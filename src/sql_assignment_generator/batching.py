'''Group incompatible error requirements into compatible batches for dataset generation.'''

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from sql_error_taxonomy import SqlErrors

from .difficulty_level import DifficultyLevel
from .error_requirements import SqlErrorRequirements


class DetailTag(Enum):
    '''Semantic tag for dataset_extra_details, used for compatibility analysis.'''
    INCONSISTENT_NAMING = auto()
    COMPLEX_NAMES = auto()
    REQUIRES_NULL = auto()
    REQUIRES_NUMERIC = auto()
    REQUIRES_DUPLICATES = auto()
    REQUIRES_EMPTY_STRINGS = auto()
    REQUIRES_STRING_ATTRS = auto()
    REQUIRES_COMPOSITE_FK = auto()
    REQUIRES_FK = auto()
    REQUIRES_DUPLICATE_GROUPBY = auto()


# Static registry: maps SqlErrors to the semantic tags of their dataset_extra_details
# and any schema constraint types they add beyond the base requirements.
ERROR_DETAIL_TAGS: dict[SqlErrors, set[DetailTag]] = {
    SqlErrors.SYNONYMS: {DetailTag.INCONSISTENT_NAMING},
    SqlErrors.UNDEFINED_COLUMN: {DetailTag.COMPLEX_NAMES},
    SqlErrors.UNDEFINED_OBJECT: {DetailTag.COMPLEX_NAMES},
    SqlErrors.MISSPELLINGS: {DetailTag.COMPLEX_NAMES},
    SqlErrors.OMITTED_QUOTES: {DetailTag.REQUIRES_STRING_ATTRS},
    SqlErrors.COMPARISON_WITH_NULL: {DetailTag.REQUIRES_NULL},
    SqlErrors.DISTINCT_IN_SUM_OR_AVG: {DetailTag.REQUIRES_NUMERIC},
    SqlErrors.MIXED_A_GREATER_THAN_0_WITH_IS_NOT_NULL_OR_EMPTY_STRING_WITH_NULL: {DetailTag.REQUIRES_EMPTY_STRINGS},
    SqlErrors.NULL_IN_IN_ANY_ALL_SUBQUERY: {DetailTag.REQUIRES_NULL},
    SqlErrors.JOIN_CONDITION_ON_INCORRECT_COLUMN: {DetailTag.REQUIRES_COMPOSITE_FK},
    SqlErrors.MISSING_DISTINCT_FROM_SELECT: {DetailTag.REQUIRES_DUPLICATES},
    SqlErrors.DISTINCT_AS_FUNCTION_PARAMETER_WHERE_NOT_APPLICABLE: {DetailTag.REQUIRES_DUPLICATES},
    SqlErrors.MISSING_DISTINCT_FROM_FUNCTION_PARAMETER: {DetailTag.REQUIRES_DUPLICATES},
    SqlErrors.UNNECESSARY_TABLE_REFERENCE: {DetailTag.REQUIRES_FK},
    SqlErrors.UNNECESSARILY_COMPLICATED_SELECT_IN_EXISTS_SUBQUERY: {DetailTag.REQUIRES_DUPLICATES},
    SqlErrors.GROUP_BY_WITH_SINGLETON_GROUPS: {DetailTag.REQUIRES_DUPLICATE_GROUPBY},
}

# Hard incompatibility rules: these tag pairs cannot coexist in the same dataset.
INCOMPATIBLE_PAIRS: set[frozenset[DetailTag]] = {
    frozenset({DetailTag.INCONSISTENT_NAMING, DetailTag.COMPLEX_NAMES}),
}

# Maximum number of unique non-NONE detail tags per batch before splitting.
MAX_UNIQUE_TAGS_PER_BATCH: int = 4


@dataclass
class ErrorProfile:
    '''Summary of an error's dataset requirements for compatibility analysis.'''

    error: SqlErrors
    difficulty: DifficultyLevel
    requirement: SqlErrorRequirements
    tags: set[DetailTag] = field(default_factory=set)


def build_profiles(
        requirements: list[tuple[SqlErrors, SqlErrorRequirements, DifficultyLevel]],
    ) -> list[ErrorProfile]:
    '''Build a profile for each error requirement, including its detail tags.'''
    profiles: list[ErrorProfile] = []

    for error, requirement, difficulty in requirements:
        tags = ERROR_DETAIL_TAGS.get(error, set()).copy()
        profiles.append(ErrorProfile(
            error=error,
            difficulty=difficulty,
            requirement=requirement,
            tags=tags,
        ))

    return profiles


def are_compatible(profiles: list[ErrorProfile]) -> bool:
    '''Check whether a group of profiles can share a single dataset.

    Returns False if any hard incompatibility exists between tags,
    or if the total number of unique tags exceeds the threshold.
    '''
    all_tags: set[DetailTag] = set()

    for profile in profiles:
        for tag in profile.tags:
            for existing in all_tags:
                if frozenset({tag, existing}) in INCOMPATIBLE_PAIRS:
                    return False
            all_tags.add(tag)

    if len(all_tags) > MAX_UNIQUE_TAGS_PER_BATCH:
        return False

    return True


def group_errors(profiles: list[ErrorProfile]) -> list[list[ErrorProfile]]:
    '''Group error profiles into compatible batches using greedy first-fit.

    Each batch contains profiles that can share a single dataset.
    '''
    batches: list[list[ErrorProfile]] = []

    for profile in profiles:
        placed = False

        for batch in batches:
            candidate = batch + [profile]
            if are_compatible(candidate):
                batch.append(profile)
                placed = True
                break

        if not placed:
            batches.append([profile])

    return batches
