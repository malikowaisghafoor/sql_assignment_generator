'''Generate SQL assignments based on specified SQL errors and difficulty levels.'''

from __future__ import annotations

from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random

from .difficulty_level import DifficultyLevel
from .domains import random_domain
from .assignments import Assignment, Dataset, Exercise
from .constraints import SchemaConstraint, QueryConstraint
from .error_requirements import SqlErrorRequirements, ERROR_REQUIREMENTS_MAP
from .exceptions import ExerciseGenerationError, SQLParsingError
from .query_executor import execute_query, validate_assignment
from .assignments.dataset import strings as dataset_strings
from .db import get_database, QueryExecutionError
from .batching import build_profiles, group_errors

import dav_tools
from sql_error_taxonomy import SqlErrors
import sqlglot


def _validate_and_fix_queries(
        assignment: Assignment,
        sql_dialect: str,
        language: str,
        max_regeneration_attempts: int,
        db_host: str,
        db_port: int,
        db_user: str,
        db_password: str,
    ) -> Assignment:
    '''Execute all exercises, regenerate data for queries returning no results.'''

    results = validate_assignment(assignment, sql_dialect, db_host, db_port, db_user, db_password)

    # Find exercises that return no results (and not due to execution errors)
    failing_indices = [
        i for i, (_, result) in enumerate(results)
        if not result.has_results and result.success
    ]

    if not failing_indices:
        return assignment

    dav_tools.messages.warning(
        f'{len(failing_indices)} exercise(s) return no results. Attempting to regenerate sample data.'
    )

    dataset = assignment.dataset
    exercises = assignment.exercises
    schema_sql = '\n'.join(dataset.create_commands)

    for idx in failing_indices:
        exercise = exercises[idx]
        query_sql = exercise.solutions[0].sql
        current_data = '\n'.join(dataset.insert_commands)

        regenerated = False

        for attempt in range(max_regeneration_attempts):
            # Ask LLM to regenerate INSERT data
            from . import llm

            prompt = dataset_strings.prompt_regenerate_data(
                schema_sql=schema_sql,
                failing_query=query_sql,
                current_data=current_data,
                sql_dialect=sql_dialect,
                language=language,
            )

            messages = llm.Message()
            messages.add_message_user(prompt)

            try:
                answer = llm.generate_answer(messages, json_format=llm.models.InsertData)
                assert isinstance(answer, llm.models.InsertData)

                # Parse and normalize new INSERT commands
                new_inserts = []
                for cmd in answer.insert_commands:
                    try:
                        parsed = sqlglot.parse_one(cmd, read=sql_dialect)
                        new_inserts.append(parsed)
                    except Exception:
                        new_inserts.append(cmd)

                # Normalize using the existing function
                from .assignments.dataset.dataset import _normalize_inserts
                new_insert_commands = _normalize_inserts(
                    [i for i in new_inserts if hasattr(i, 'sql')],
                    sql_dialect
                )

                if not new_insert_commands:
                    continue

                # Try the new dataset
                new_dataset = dataset.with_inserts(new_insert_commands)

                # Verify against the real DBMS
                all_pass = False
                with get_database(db_host, db_port, db_user, db_password, sql_dialect) as db:
                    try:
                        db.execute(new_dataset.to_sql_no_context())
                    except QueryExecutionError:
                        dav_tools.messages.warning(
                            f'{exercise.title}: Regenerated data failed to load (attempt {attempt + 1}).'
                        )
                        continue

                    # Verify the failing query now returns results
                    failing_result = execute_query(db, query_sql)
                    if not failing_result.has_results:
                        dav_tools.messages.warning(
                            f'{exercise.title}: Regenerated data still returns no results (attempt {attempt + 1}).'
                        )
                        continue

                    # Verify ALL other exercises still work
                    all_pass = True
                    for other_idx, other_exercise in enumerate(exercises):
                        if other_idx == idx:
                            continue
                        other_result = execute_query(db, other_exercise.solutions[0].sql)
                        if other_result.success and not other_result.has_results:
                            all_pass = False
                            dav_tools.messages.warning(
                                f'{other_exercise.title}: Broken by regenerated data.'
                            )
                            break

                if all_pass:
                    dataset = new_dataset
                    regenerated = True
                    dav_tools.messages.success(
                        f'{exercise.title}: Data regenerated successfully.'
                    )
                    break
                else:
                    dav_tools.messages.warning(
                        f'{exercise.title}: Regenerated data broke other exercises (attempt {attempt + 1}).'
                    )

            except Exception as e:
                dav_tools.messages.error(
                    f'{exercise.title}: Error regenerating data (attempt {attempt + 1}): {e}'
                )

        if not regenerated:
            dav_tools.messages.warning(
                f'{exercise.title}: Could not regenerate data after {max_regeneration_attempts} attempts. Keeping original data.'
            )

    return Assignment(dataset=dataset, exercises=exercises)


def _generate_batch(
        requirements: list[tuple[SqlErrors, SqlErrorRequirements, DifficultyLevel]],
        db_host: str,
        db_port: int,
        db_user: str,
        db_password: str,
        sql_dialect: str,
        language: str,
        domain: str | None,
        dataset_str: str | None,
        naming_func: Callable[[SqlErrors, DifficultyLevel], str],
        max_dataset_attempts: int,
        max_exercise_attempts: int,
        max_unique_attempts: int,
        max_workers: int | None,
        validate_queries: bool,
        max_regeneration_attempts: int,
    ) -> Assignment:
    '''Generate a single assignment for one batch of compatible error requirements.'''

    if domain is None:
        domain = random_domain(language=language)

    if not dataset_str:
        dataset_requirements: list[SchemaConstraint] = []
        for _, req, difficulty in requirements:
            dataset_requirements.extend(req.dataset_constraints(difficulty))

        dataset_extra_details: list[str] = [
            req.dataset_extra_details().get(language=language)
            for _, req, _ in requirements
        ]
        dataset_extra_details = [detail for detail in dataset_extra_details if detail.strip()]
        dataset_extra_details = list(set(dataset_extra_details))

        dav_tools.messages.info(f'Generating dataset for domain: {domain}')
        dataset = Dataset.generate(
            domain=domain,
            sql_dialect=sql_dialect,
            constraints=dataset_requirements,
            extra_details=dataset_extra_details,
            language=language,
            max_attempts=max_dataset_attempts,
            db_host=db_host,
            db_port=db_port,
            db_user=db_user,
            db_password=db_password,
        )
        dav_tools.messages.success('Dataset generated')
    else:
        dataset = Dataset.from_sql(
            sql_str=dataset_str,
            sql_dialect=sql_dialect,
        )

    generated_solutions_hashes: set[str] = set()
    hashes_lock = threading.Lock()
    log_lock = threading.Lock()

    def _worker(
            idx: int,
            error: SqlErrors,
            difficulty: DifficultyLevel,
            constraints: list[QueryConstraint],
            extra_details: str,
    ) -> tuple[int, Exercise | None]:
        title = naming_func(error, difficulty)
        dav_tools.messages.info(f'Starting generation for exercise: {title}')

        last_generated_exercise: Exercise | None = None

        for attempt in range(max_unique_attempts):
            try:
                generated_exercise = Exercise.generate(
                    error=error,
                    difficulty=difficulty,
                    constraints=constraints,
                    extra_details=extra_details,
                    sql_dialect=sql_dialect,
                    dataset=dataset,
                    title=title,
                    max_attempts=max_exercise_attempts,
                    language=language,
                    db_host=db_host,
                    db_port=db_port,
                    db_user=db_user,
                    db_password=db_password,
                )
            except ExerciseGenerationError:
                with log_lock:
                    dav_tools.messages.warning(f'{title}: Skipping exercise generation for {error.name} due to validation failures.')
                return (idx, None)

            last_generated_exercise = generated_exercise
            raw_solution = generated_exercise.solutions[0]
            normalized_solution = raw_solution.sql.lower().strip()

            with hashes_lock:
                is_duplicate = normalized_solution in generated_solutions_hashes
                if not is_duplicate:
                    generated_solutions_hashes.add(normalized_solution)

            if is_duplicate:
                with log_lock:
                    dav_tools.messages.warning(f'{title}: Duplicate solution detected for {error.name} (Attempt {attempt + 1}/{max_unique_attempts}). Regenerating...')
                continue

            with log_lock:
                dav_tools.messages.info(f'{title}: Successfully generated.')

            return (idx, generated_exercise)

        if last_generated_exercise is not None:
            with log_lock:
                dav_tools.messages.error(f'{title}: Could not generate a UNIQUE exercise for {error.name} after {max_unique_attempts} retries. Skipping.')
        return (idx, None)

    ordered_results: list[Exercise | None] = [None] * len(requirements)

    if max_workers == 1:
        for idx, (error, requirement, difficulty) in enumerate(requirements):
            i, ex = _worker(
                idx=idx,
                error=error,
                difficulty=difficulty,
                constraints=requirement.exercise_constraints(difficulty),
                extra_details=requirement.exercise_extra_details().get(language=language),
            )
            ordered_results[i] = ex
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _worker,
                    idx=idx,
                    error=error,
                    difficulty=difficulty,
                    constraints=requirement.exercise_constraints(difficulty),
                    extra_details=requirement.exercise_extra_details().get(language=language),
                )
                for idx, (error, requirement, difficulty) in enumerate(requirements)
            ]
            for fut in as_completed(futures):
                idx, ex = fut.result()
                ordered_results[idx] = ex

    exercises: list[Exercise] = [ex for ex in ordered_results if ex is not None]

    if len(exercises) < len(requirements):
        dav_tools.messages.warning(f'Finished generating exercises with some failures. Generated: {len(exercises)}. Failed: {len(requirements) - len(exercises)}.')
    else:
        dav_tools.messages.success(f'Successfully generated all {len(exercises)} exercises.')

    assignment = Assignment(dataset=dataset, exercises=exercises)

    if validate_queries and exercises:
        assignment = _validate_and_fix_queries(
            assignment=assignment,
            sql_dialect=sql_dialect,
            language=language,
            max_regeneration_attempts=max_regeneration_attempts,
            db_host=db_host,
            db_port=db_port,
            db_user=db_user,
            db_password=db_password,
        )

    return assignment


def generate_assignments(
        errors: list[tuple[SqlErrors, DifficultyLevel]],
        db_host: str,
        db_port: int,
        db_user: str,
        db_password: str,
        sql_dialect: str = 'postgres',
        *,
        language: str = 'en',
        domain: str | None = None,
        dataset_str: str | None = None,
        shuffle_exercises: bool = False,
        naming_func: Callable[[SqlErrors, DifficultyLevel], str] = lambda error, difficulty: f'{error.name} - {difficulty.name}',
        max_dataset_attempts: int = 3,
        max_exercise_attempts: int = 3,
        max_unique_attempts: int = 3,
        max_workers: int | None = None,
        validate_queries: bool = True,
        max_regeneration_attempts: int = 2,
    ) -> list[Assignment]:
    '''
    Generate SQL assignments, automatically batching incompatible constraints
    into separate datasets. Each Assignment has its own Dataset and exercises.

    Returns one Assignment per compatible batch of errors.
    '''
    # filter only supported errors
    supported_errors: list[tuple[SqlErrors, DifficultyLevel]] = []
    for error, difficulty in errors:
        if error in ERROR_REQUIREMENTS_MAP:
            supported_errors.append((error, difficulty))
        else:
            dav_tools.messages.warning(f'Skipping unsupported error: {error.name}')

    if not supported_errors:
        raise ValueError('No supported errors provided for assignment generation.')

    if shuffle_exercises:
        random.shuffle(supported_errors)

    dav_tools.messages.info(f'Starting assignment generation for {len(supported_errors)} exercises (out of {len(errors)} requested)')

    # convert SqlErrors -> SqlErrorRequirements
    requirements: list[tuple[SqlErrors, SqlErrorRequirements, DifficultyLevel]] = [
        (error, ERROR_REQUIREMENTS_MAP[error](language=language), difficulty)
        for error, difficulty in supported_errors
    ]

    # If a dataset_str is provided, all errors share it — no batching needed
    if dataset_str:
        return [_generate_batch(
            requirements=requirements,
            db_host=db_host, db_port=db_port, db_user=db_user, db_password=db_password,
            sql_dialect=sql_dialect, language=language, domain=domain,
            dataset_str=dataset_str, naming_func=naming_func,
            max_dataset_attempts=max_dataset_attempts, max_exercise_attempts=max_exercise_attempts,
            max_unique_attempts=max_unique_attempts, max_workers=max_workers,
            validate_queries=validate_queries, max_regeneration_attempts=max_regeneration_attempts,
        )]

    # Build profiles and group into compatible batches
    profiles = build_profiles(requirements)
    batches = group_errors(profiles)

    if len(batches) > 1:
        dav_tools.messages.info(f'Detected incompatible constraints. Split into {len(batches)} batches.')

    assignments: list[Assignment] = []
    for batch_profiles in batches:
        batch_requirements = [
            (p.error, p.requirement, p.difficulty)
            for p in batch_profiles
        ]
        assignment = _generate_batch(
            requirements=batch_requirements,
            db_host=db_host, db_port=db_port, db_user=db_user, db_password=db_password,
            sql_dialect=sql_dialect, language=language, domain=domain,
            dataset_str=None, naming_func=naming_func,
            max_dataset_attempts=max_dataset_attempts, max_exercise_attempts=max_exercise_attempts,
            max_unique_attempts=max_unique_attempts, max_workers=max_workers,
            validate_queries=validate_queries, max_regeneration_attempts=max_regeneration_attempts,
        )
        assignments.append(assignment)

    return assignments


def generate_assignment(
        errors: list[tuple[SqlErrors, DifficultyLevel]],
        db_host: str,
        db_port: int,
        db_user: str,
        db_password: str,
        sql_dialect: str = 'postgres',
        *,
        language: str = 'en',
        domain: str | None = None,
        dataset_str: str | None = None,
        shuffle_exercises: bool = False,
        naming_func: Callable[[SqlErrors, DifficultyLevel], str] = lambda error, difficulty: f'{error.name} - {difficulty.name}',
        max_dataset_attempts: int = 3,
        max_exercise_attempts: int = 3,
        max_unique_attempts: int = 3,
        max_workers: int | None = None,
        validate_queries: bool = True,
        max_regeneration_attempts: int = 2,
    ) -> Assignment:
    '''
    Generate a SQL assignment based on the given SQL errors and their corresponding difficulty levels.

    Automatically batches incompatible constraints into separate datasets and merges the results.

    Args:
        errors (list[tuple[SqlErrors, DifficultyLevel]]): A list of (error, difficulty) pairs.
        sql_dialect (str): The SQL dialect to use (e.g., 'postgres', 'mysql').
        db_host (str): Database host.
        db_port (int): Database port.
        db_user (str): Database user.
        db_password (str): Database password.
        language (str): Language for generation (e.g., 'en', 'it').
        domain (str | None): Domain for the dataset. If None, a random domain is selected.
        dataset_str (str | None): Optional SQL string to use as the dataset instead of generating one.
        shuffle_exercises (bool): Whether to shuffle exercises to prevent ordering bias.
        naming_func (Callable): Generates exercise titles.
        max_dataset_attempts (int): Maximum retries for dataset generation.
        max_exercise_attempts (int): Maximum retries for exercise generation.
        max_unique_attempts (int): Maximum retries to avoid duplicate solutions.
        max_workers (int | None): Thread pool size.
        validate_queries (bool): Whether to validate queries by execution.
        max_regeneration_attempts (int): Maximum retries for data regeneration.

    Returns:
        Assignment: The generated assignment.
    '''
    assignments = generate_assignments(
        errors=errors,
        db_host=db_host, db_port=db_port, db_user=db_user, db_password=db_password,
        sql_dialect=sql_dialect,
        language=language, domain=domain, dataset_str=dataset_str,
        shuffle_exercises=shuffle_exercises, naming_func=naming_func,
        max_dataset_attempts=max_dataset_attempts, max_exercise_attempts=max_exercise_attempts,
        max_unique_attempts=max_unique_attempts, max_workers=max_workers,
        validate_queries=validate_queries, max_regeneration_attempts=max_regeneration_attempts,
    )

    if len(assignments) == 1:
        return assignments[0]

    # Multiple batches: merge into a single Assignment
    dav_tools.messages.warning(
        f'Incompatible constraints detected — generated {len(assignments)} separate datasets. '
        f'Merging into a single assignment.'
    )

    all_create_commands: list[str] = []
    all_insert_commands: list[str] = []
    all_exercises: list[Exercise] = []

    for assignment in assignments:
        all_create_commands.extend(assignment.dataset.create_commands)
        all_insert_commands.extend(assignment.dataset.insert_commands)
        all_exercises.extend(assignment.exercises)

    merged_dataset = Dataset(
        create_commands=all_create_commands,
        insert_commands=all_insert_commands,
        domain=assignments[0].dataset.domain,
    )

    return Assignment(dataset=merged_dataset, exercises=all_exercises)
