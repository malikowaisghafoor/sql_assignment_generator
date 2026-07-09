'''Tests for the Spider hardness metric (Goal 1).'''

import pytest
from sqlscope import Query

from sql_assignment_generator.complexity.base import ComplexityLevel
from sql_assignment_generator.complexity import spider
from sql_assignment_generator.complexity.spider import SpiderCounts


# =================================================================
# DECISION-TREE UNIT TESTS (pure, no parsing)
# =================================================================

@pytest.mark.parametrize("counts, expected", [
    # Easy: 0 or 1 from components1, no others, no components2
    (SpiderCounts(component1=0, component2=0, others=0), ComplexityLevel.EASY),
    (SpiderCounts(component1=1, component2=0, others=0), ComplexityLevel.EASY),
    # Medium A: others <= 2, c1 <= 1, c2 == 0
    (SpiderCounts(component1=1, component2=0, others=2), ComplexityLevel.MEDIUM),
    # Medium B: c1 == 2, others < 2, c2 == 0
    (SpiderCounts(component1=2, component2=0, others=1), ComplexityLevel.MEDIUM),
    # Hard A: others > 2, c1 <= 2, c2 == 0
    (SpiderCounts(component1=2, component2=0, others=3), ComplexityLevel.HARD),
    # Hard B: 2 < c1 <= 3, others <= 2, c2 == 0
    (SpiderCounts(component1=3, component2=0, others=2), ComplexityLevel.HARD),
    # Hard C: c1 <= 1, others == 0, c2 == 1
    (SpiderCounts(component1=1, component2=1, others=0), ComplexityLevel.HARD),
    # Extra: c2 >= 2
    (SpiderCounts(component1=0, component2=2, others=0), ComplexityLevel.EXTRA),
    # Extra: c1 == 4
    (SpiderCounts(component1=4, component2=0, others=0), ComplexityLevel.EXTRA),
])
def test_spider_hardness_decision_tree(counts, expected):
    assert spider.spider_hardness(counts) == expected


# =================================================================
# COUNTING + END-TO-END ON SQL
# =================================================================

@pytest.mark.parametrize("sql, expected", [
    # Easy: single keyword (WHERE), no others, no c2
    ("SELECT a FROM t WHERE a = 1", ComplexityLevel.EASY),
    # Easy: zero keywords
    ("SELECT a FROM t", ComplexityLevel.EASY),
    # Medium: 2 keywords from c1 (WHERE + ORDER BY)
    ("SELECT a FROM t WHERE a > 1 ORDER BY a", ComplexityLevel.MEDIUM),
    # Hard: a single set operator with nothing else (c1<=1, others==0, c2==1)
    ("SELECT a FROM t UNION SELECT a FROM t2", ComplexityLevel.HARD),
])
def test_spider_evaluate_sql(sql, expected):
    assert spider.evaluate_sql(sql) == expected


def test_spider_counts_basic_query():
    counts = spider.count_spider(Query("SELECT a, b FROM t WHERE a = 1 AND b = 2"))
    # WHERE present, 2 selected columns, 2 where conditions
    assert counts.where == 1
    assert counts.select_columns == 2
    assert counts.where_conditions == 2
    assert counts.component1 >= 1
    assert counts.others >= 2  # select_columns > 1 and where_conditions > 1


def test_spider_counts_join_and_aggregation():
    sql = "SELECT COUNT(*) FROM t1 JOIN t2 ON t1.id = t2.id GROUP BY t1.x"
    counts = spider.count_spider(Query(sql))
    assert counts.join == 1
    assert counts.group_by == 1
    assert counts.agg_count == 1