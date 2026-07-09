'''Tests for the cognitive (construct-weighted) complexity metric (Goal 1).'''

import pytest
from sqlscope import Query

from sql_assignment_generator.complexity.base import ComplexityLevel
from sql_assignment_generator.complexity import cognitive
from sql_assignment_generator.complexity.cognitive import Weights, Thresholds


def test_cognitive_default_weights_select():
    # SELECT only -> weight 1 -> score 1 -> EASY
    score, bd = cognitive.cognitive_score(Query("SELECT a FROM t"))
    assert score == 1
    assert bd.select == 1
    assert cognitive.cognitive_hardness(score) == ComplexityLevel.EASY


def test_cognitive_join_weight():
    # SELECT(1) + JOIN(2) = 3
    score, bd = cognitive.cognitive_score(Query("SELECT a FROM t1 JOIN t2 ON t1.id = t2.id"))
    assert bd.join == 2
    assert score == 3
    assert cognitive.cognitive_hardness(score) == ComplexityLevel.MEDIUM


def test_cognitive_subquery_weight():
    # Outer SELECT(1) + WHERE(1) + inner SELECT(1) + subquery(4) = 7.
    # (Each SELECT construct contributes 1, and the IN-subquery is a non-
    # correlated subquery of weight 4.)
    score, bd = cognitive.cognitive_score(Query("SELECT a FROM t WHERE id IN (SELECT id FROM t2)"))
    assert bd.subquery == 4
    assert bd.correlated_subquery == 0
    assert score == 7


def test_cognitive_custom_weights():
    # Giovanna-style custom weights: JOIN=5
    w = Weights(join=5)
    score, bd = cognitive.cognitive_score(
        Query("SELECT a FROM t1 JOIN t2 ON t1.id = t2.id"),
        weights=w,
    )
    assert bd.join == 5
    assert score == 6  # select(1) + join(5)


def test_cognitive_custom_thresholds():
    score, _ = cognitive.cognitive_score(Query("SELECT a FROM t1 JOIN t2 ON t1.id = t2.id"))
    # Default would be MEDIUM; raise the easy ceiling to make it EASY
    assert cognitive.cognitive_hardness(score, thresholds=Thresholds(easy_max=3)) == ComplexityLevel.EASY


def test_cognitive_having_weight():
    sql = "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1"
    score, bd = cognitive.cognitive_score(Query(sql))
    # SELECT(1) + GROUP BY(2) + HAVING(3) = 6
    assert bd.group_by == 2
    assert bd.having == 3
    assert score == 6