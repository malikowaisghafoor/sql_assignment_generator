import pytest
from sql_error_taxonomy import SqlErrors
from sql_assignment_generator.batching import (
    DetailTag,
    ERROR_DETAIL_TAGS,
    INCOMPATIBLE_PAIRS,
    ErrorProfile,
    build_profiles,
    are_compatible,
    group_errors,
)
from sql_assignment_generator.difficulty_level import DifficultyLevel
from sql_assignment_generator.error_requirements import ERROR_REQUIREMENTS_MAP


def _make_profile(error: SqlErrors, difficulty: DifficultyLevel = DifficultyLevel.EASY) -> ErrorProfile:
    '''Build an ErrorProfile for testing.'''
    req_class = ERROR_REQUIREMENTS_MAP[error]
    req = req_class(language='en')
    return ErrorProfile(
        error=error,
        difficulty=difficulty,
        requirement=req,
        tags=ERROR_DETAIL_TAGS.get(error, set()).copy(),
    )


# =================================================================
# DETAIL TAG REGISTRY
# =================================================================

class TestDetailTagRegistry:

    def test_err_010_has_inconsistent_naming(self):
        assert DetailTag.INCONSISTENT_NAMING in ERROR_DETAIL_TAGS[SqlErrors.SYNONYMS]

    def test_err_004_has_complex_names(self):
        assert DetailTag.COMPLEX_NAMES in ERROR_DETAIL_TAGS[SqlErrors.UNDEFINED_COLUMN]

    def test_err_007_has_complex_names(self):
        assert DetailTag.COMPLEX_NAMES in ERROR_DETAIL_TAGS[SqlErrors.UNDEFINED_OBJECT]

    def test_err_009_has_complex_names(self):
        assert DetailTag.COMPLEX_NAMES in ERROR_DETAIL_TAGS[SqlErrors.MISSPELLINGS]

    def test_err_021_requires_null(self):
        assert DetailTag.REQUIRES_NULL in ERROR_DETAIL_TAGS[SqlErrors.COMPARISON_WITH_NULL]

    def test_err_072_requires_duplicates(self):
        assert DetailTag.REQUIRES_DUPLICATES in ERROR_DETAIL_TAGS[SqlErrors.MISSING_DISTINCT_FROM_SELECT]

    def test_err_095_requires_duplicate_groupby(self):
        assert DetailTag.REQUIRES_DUPLICATE_GROUPBY in ERROR_DETAIL_TAGS[SqlErrors.GROUP_BY_WITH_SINGLETON_GROUPS]

    def test_unlisted_error_has_no_tags(self):
        # err_002 is not in the registry
        assert SqlErrors.AMBIGUOUS_COLUMN not in ERROR_DETAIL_TAGS


# =================================================================
# ARE COMPATIBLE
# =================================================================

class TestAreCompatible:

    def test_inconsistent_naming_vs_complex_names(self):
        p1 = ErrorProfile(error=SqlErrors.SYNONYMS, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.INCONSISTENT_NAMING})
        p2 = ErrorProfile(error=SqlErrors.UNDEFINED_COLUMN, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.COMPLEX_NAMES})
        assert not are_compatible([p1, p2])

    def test_compatible_errors_no_tags(self):
        p1 = ErrorProfile(error=SqlErrors.AMBIGUOUS_COLUMN, difficulty=DifficultyLevel.EASY, requirement=None, tags=set())
        p2 = ErrorProfile(error=SqlErrors.OMITTING_CORRELATION_NAMES, difficulty=DifficultyLevel.EASY, requirement=None, tags=set())
        assert are_compatible([p1, p2])

    def test_compatible_duplicate_errors(self):
        p1 = ErrorProfile(error=SqlErrors.MISSING_DISTINCT_FROM_SELECT, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.REQUIRES_DUPLICATES})
        p2 = ErrorProfile(error=SqlErrors.DISTINCT_AS_FUNCTION_PARAMETER_WHERE_NOT_APPLICABLE, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.REQUIRES_DUPLICATES})
        assert are_compatible([p1, p2])

    def test_single_profile_always_compatible(self):
        p = ErrorProfile(error=SqlErrors.SYNONYMS, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.INCONSISTENT_NAMING})
        assert are_compatible([p])

    def test_empty_list_compatible(self):
        assert are_compatible([])

    def test_tag_overload_exceeds_threshold(self):
        # 5 unique tags exceeds MAX_UNIQUE_TAGS_PER_BATCH=4
        profiles = [
            ErrorProfile(error=None, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.REQUIRES_NULL}),
            ErrorProfile(error=None, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.REQUIRES_NUMERIC}),
            ErrorProfile(error=None, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.REQUIRES_DUPLICATES}),
            ErrorProfile(error=None, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.REQUIRES_FK}),
            ErrorProfile(error=None, difficulty=DifficultyLevel.EASY, requirement=None, tags={DetailTag.REQUIRES_STRING_ATTRS}),
        ]
        assert not are_compatible(profiles)


# =================================================================
# GROUP ERRORS
# =================================================================

class TestGroupErrors:

    def test_incompatible_split_into_two_batches(self):
        p1 = _make_profile(SqlErrors.SYNONYMS)  # INCONSISTENT_NAMING
        p2 = _make_profile(SqlErrors.UNDEFINED_COLUMN)  # COMPLEX_NAMES
        batches = group_errors([p1, p2])
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1

    def test_compatible_stay_in_one_batch(self):
        p1 = _make_profile(SqlErrors.MISSING_DISTINCT_FROM_SELECT)  # REQUIRES_DUPLICATES
        p2 = _make_profile(SqlErrors.DISTINCT_AS_FUNCTION_PARAMETER_WHERE_NOT_APPLICABLE)  # REQUIRES_DUPLICATES
        batches = group_errors([p1, p2])
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_mixed_compatible_and_incompatible(self):
        p1 = _make_profile(SqlErrors.SYNONYMS)  # INCONSISTENT_NAMING
        p2 = _make_profile(SqlErrors.UNDEFINED_COLUMN)  # COMPLEX_NAMES
        p3 = _make_profile(SqlErrors.AMBIGUOUS_COLUMN)  # no tags
        batches = group_errors([p1, p2, p3])
        # p1 and p3 are compatible, p2 conflicts with p1
        # Greedy: p1 starts batch1, p2 can't join → new batch2, p3 can join batch1
        assert len(batches) == 2
        errors_batch1 = [p.error for p in batches[0]]
        assert SqlErrors.SYNONYMS in errors_batch1
        assert SqlErrors.AMBIGUOUS_COLUMN in errors_batch1
        assert batches[1][0].error == SqlErrors.UNDEFINED_COLUMN

    def test_single_error_one_batch(self):
        p = _make_profile(SqlErrors.AMBIGUOUS_COLUMN)
        batches = group_errors([p])
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_empty_input(self):
        batches = group_errors([])
        assert len(batches) == 0

    def test_preserves_input_order(self):
        errors = [SqlErrors.AMBIGUOUS_COLUMN, SqlErrors.MISSING_DISTINCT_FROM_SELECT]
        profiles = [_make_profile(e) for e in errors]
        batches = group_errors(profiles)
        batch_errors = [p.error for p in batches[0]]
        assert batch_errors == errors


# =================================================================
# BUILD PROFILES
# =================================================================

class TestBuildProfiles:

    def test_builds_correct_profiles(self):
        req_class = ERROR_REQUIREMENTS_MAP[SqlErrors.SYNONYMS]
        requirements = [
            (SqlErrors.SYNONYMS, req_class(language='en'), DifficultyLevel.EASY),
        ]
        profiles = build_profiles(requirements)
        assert len(profiles) == 1
        assert profiles[0].error == SqlErrors.SYNONYMS
        assert DetailTag.INCONSISTENT_NAMING in profiles[0].tags

    def test_unlisted_error_gets_empty_tags(self):
        req_class = ERROR_REQUIREMENTS_MAP[SqlErrors.AMBIGUOUS_COLUMN]
        requirements = [
            (SqlErrors.AMBIGUOUS_COLUMN, req_class(language='en'), DifficultyLevel.EASY),
        ]
        profiles = build_profiles(requirements)
        assert len(profiles) == 1
        assert profiles[0].tags == set()
