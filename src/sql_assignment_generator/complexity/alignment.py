'''Cross-metric alignment helpers (Goal 1).

``goals.txt`` explicitly asks to *"Check whether they are aligned"* — i.e.
compare the labels produced by the Spider, cognitive and structural metrics on
the same set of queries.  This module provides small utilities to run all
metrics side-by-side and summarise their agreement.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from sqlscope import Query

from . import cognitive, spider, structural
from .base import ComplexityLevel

__all__ = ['MetricResult', 'compare_query', 'compare_queries', 'agreement_matrix', 'AgreementReport']


# A metric is any callable ``Query -> ComplexityLevel``.
Metric = Callable[[Query], ComplexityLevel]


@dataclass(frozen=True)
class MetricResult:
    '''The labels assigned by each metric to a single query.'''

    sql: str
    spider: ComplexityLevel
    cognitive: ComplexityLevel
    structural: ComplexityLevel


def _default_metrics() -> dict[str, Metric]:
    return {
        'spider': spider.evaluate,
        'cognitive': cognitive.evaluate,
        'structural': structural.evaluate,
    }


def compare_query(
    query: Query,
    *,
    metrics: dict[str, Metric] | None = None,
) -> dict[str, ComplexityLevel]:
    '''Run every metric on a single ``Query`` and return ``{name: level}``.'''
    metrics = metrics or _default_metrics()
    return {name: fn(query) for name, fn in metrics.items()}


def compare_queries(
    queries: Iterable[Query | str],
    *,
    metrics: dict[str, Metric] | None = None,
) -> list[MetricResult]:
    '''Run all metrics on many queries, returning :class:`MetricResult` rows.

    Accepts either pre-parsed ``Query`` objects or raw SQL strings.
    '''
    from sqlscope import Query as _Query

    metrics = metrics or _default_metrics()
    results: list[MetricResult] = []

    for item in queries:
        if isinstance(item, str):
            query = _Query(item)
            sql = item
        else:
            query = item
            sql = query.sql

        labels = {name: fn(query) for name, fn in metrics.items()}
        results.append(
            MetricResult(
                sql=sql,
                spider=labels.get('spider', ComplexityLevel.EASY),
                cognitive=labels.get('cognitive', ComplexityLevel.EASY),
                structural=labels.get('structural', ComplexityLevel.EASY),
            )
        )

    return results


@dataclass
class AgreementReport:
    '''Summary of how often two metrics agree.'''

    metric_a: str
    metric_b: str
    total: int
    agreements: int

    @property
    def agreement_rate(self) -> float:
        return self.agreements / self.total if self.total else 0.0

    def __str__(self) -> str:
        pct = self.agreement_rate * 100
        return f'{self.metric_a} vs {self.metric_b}: {self.agreements}/{self.total} ({pct:.1f}%)'


def agreement_matrix(results: list[MetricResult]) -> list[AgreementReport]:
    '''Pairwise agreement between the three metrics over ``results``.'''
    pairs = [('spider', 'cognitive'), ('spider', 'structural'), ('cognitive', 'structural')]
    reports: list[AgreementReport] = []

    for a, b in pairs:
        total = len(results)
        agreements = sum(1 for r in results if getattr(r, a) == getattr(r, b))
        reports.append(AgreementReport(metric_a=a, metric_b=b, total=total, agreements=agreements))

    return reports