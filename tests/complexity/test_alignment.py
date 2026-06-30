'''Tests for the cross-metric alignment helpers (Goal 1).'''

from sql_assignment_generator.complexity import alignment
from sql_assignment_generator.complexity.base import ComplexityLevel


SAMPLES = [
    "SELECT a FROM t",
    "SELECT a FROM t WHERE a = 1",
    "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id",
    "SELECT a FROM t WHERE id IN (SELECT id FROM t2)",
    "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1",
    "SELECT a FROM t UNION SELECT a FROM t2",
]


def test_compare_query_returns_all_metrics():
    from sqlscope import Query
    labels = alignment.compare_query(Query(SAMPLES[0]))
    assert set(labels.keys()) == {'spider', 'cognitive', 'structural'}


def test_compare_queries_accepts_strings_and_queries():
    from sqlscope import Query
    mixed = [SAMPLES[0], Query(SAMPLES[1])]
    results = alignment.compare_queries(mixed)
    assert len(results) == 2
    assert all(r.spider in ComplexityLevel for r in results)


def test_agreement_matrix_counts():
    results = alignment.compare_queries(SAMPLES)
    reports = alignment.agreement_matrix(results)
    assert len(reports) == 3  # three pairs
    for r in reports:
        assert 0 <= r.agreements <= r.total == len(SAMPLES)
        assert 0.0 <= r.agreement_rate <= 1.0