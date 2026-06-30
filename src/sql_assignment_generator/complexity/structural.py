'''Structural complexity metric (Goal 1).

Captures the "currently used metric" described in ``goals.txt`` under
"Structural Complexity": a weighted combination of structural features such as
the number of tables, joins, subqueries, nesting depth, predicates,
aggregations and the presence of GROUP BY / HAVING / ORDER BY / set operators.

This is intended both as a baseline ("currently used metric") and as a
receptacle for the weighted structural score mentioned in the same section of
``goals.txt``.  Weights are configurable so the same implementation can host
the project's existing heuristic without code duplication.
'''

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlscope import Query

from .base import ComplexityLevel

__all__ = ['StructWeights', 'StructThresholds', 'StructBreakdown', 'structural_score', 'structural_hardness', 'evaluate', 'evaluate_sql']


@dataclass(frozen=True)
class StructWeights:
    '''Default weights reproduce a typical educational structural score.'''

    tables: int = 1
    joins: int = 2
    subqueries: int = 3
    nesting_depth: int = 3
    predicates: int = 1
    aggregations: int = 1
    group_by: int = 2
    having: int = 3
    order_by: int = 1
    set_operators: int = 3


@dataclass(frozen=True)
class StructThresholds:
    easy_max: int = 3
    medium_max: int = 7
    hard_max: int = 12


_AGG_FUNCS: tuple[type[exp.Expression], ...] = (
    exp.Avg, exp.Count, exp.Sum, exp.Max, exp.Min,
)


@dataclass
class StructBreakdown:
    tables: int = 0
    joins: int = 0
    subqueries: int = 0
    nesting_depth: int = 0
    predicates: int = 0
    aggregations: int = 0
    group_by: int = 0
    having: int = 0
    order_by: int = 0
    set_operators: int = 0

    def weighted_score(self, w: StructWeights) -> int:
        return (
            self.tables * w.tables
            + self.joins * w.joins
            + self.subqueries * w.subqueries
            + self.nesting_depth * w.nesting_depth
            + self.predicates * w.predicates
            + self.aggregations * w.aggregations
            + self.group_by * w.group_by
            + self.having * w.having
            + self.order_by * w.order_by
            + self.set_operators * w.set_operators
        )

    def as_dict(self) -> dict[str, int]:
        return {
            'tables': self.tables,
            'joins': self.joins,
            'subqueries': self.subqueries,
            'nesting_depth': self.nesting_depth,
            'predicates': self.predicates,
            'aggregations': self.aggregations,
            'group_by': self.group_by,
            'having': self.having,
            'order_by': self.order_by,
            'set_operators': self.set_operators,
        }


def _count_predicates(where: exp.Expression | None) -> int:
    if where is None:
        return 0
    return 1 + len(list(where.find_all(exp.And))) + len(list(where.find_all(exp.Or)))


def structural_score(query: Query, *, weights: StructWeights | None = None) -> tuple[int, StructBreakdown]:
    '''Compute the structural complexity score of ``query``.'''
    weights = weights or StructWeights()
    bd = StructBreakdown()

    for select in query.selects:
        ast = select.ast
        if ast is None:
            continue

        bd.tables += len(select.referenced_tables)
        bd.joins += len(list(ast.find_all(exp.Join)))
        bd.subqueries += len(select.subqueries)
        bd.nesting_depth = max(bd.nesting_depth, max((d for _, _, d in select.subqueries), default=0))
        bd.predicates += _count_predicates(select.where)
        bd.aggregations += len(list(ast.find_all(_AGG_FUNCS)))
        if len(select.group_by) > 0:
            bd.group_by += 1
        if select.having is not None:
            bd.having += 1
        if len(select.order_by) > 0:
            bd.order_by += 1
        bd.set_operators += (
            len(list(ast.find_all(exp.Union)))
            + len(list(ast.find_all(exp.Intersect)))
            + len(list(ast.find_all(exp.Except)))
        )

    return bd.weighted_score(weights), bd


def structural_hardness(score: int, *, thresholds: StructThresholds | None = None) -> ComplexityLevel:
    thresholds = thresholds or StructThresholds()
    if score <= thresholds.easy_max:
        return ComplexityLevel.EASY
    if score <= thresholds.medium_max:
        return ComplexityLevel.MEDIUM
    if score <= thresholds.hard_max:
        return ComplexityLevel.HARD
    return ComplexityLevel.EXTRA


def evaluate(query: Query, *, weights: StructWeights | None = None, thresholds: StructThresholds | None = None) -> ComplexityLevel:
    score, _ = structural_score(query, weights=weights)
    return structural_hardness(score, thresholds=thresholds)


def evaluate_sql(sql: str, *, dialect: str | None = None, weights: StructWeights | None = None, thresholds: StructThresholds | None = None) -> ComplexityLevel:
    parsed = sqlglot.parse_one(sql, dialect=dialect) if dialect else sqlglot.parse_one(sql)
    return evaluate(Query(parsed.sql()), weights=weights, thresholds=thresholds)