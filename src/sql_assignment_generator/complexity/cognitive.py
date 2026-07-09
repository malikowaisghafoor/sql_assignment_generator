'''Construct-weighted (cognitive) complexity metric (Goal 1).

This metric assigns a difficulty weight to each SQL construct appearing in the
query and sums them, as described in ``goals.txt`` ("Cognitive Complexity"):

    Construct            Weight
    -----------------    ------
    SELECT               1
    WHERE                1
    JOIN                 2
    GROUP BY             2
    HAVING               3
    Subquery             4
    Correlated subquery  5

The aggregate "cognitive score" is then bucketed into the four
:class:`ComplexityLevel` values using configurable thresholds (so the same
metric can be aligned with Spider labels if desired, as required by the
"Check whether they are aligned" task in ``goals.txt``).

Weights are configurable (the ``goals.txt`` "metric keeping into account
constructs (input by Giovanna)" use-case): pass a custom ``Weights`` to
:func:`evaluate` / :func:`cognitive_score`.
'''

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlscope import Query

from .base import ComplexityLevel

__all__ = ['Weights', 'Thresholds', 'CognitiveBreakdown', 'cognitive_score', 'cognitive_hardness', 'evaluate', 'evaluate_sql']


@dataclass(frozen=True)
class Weights:
    '''Per-construct difficulty weights.

    The defaults reproduce the table in ``goals.txt``.
    '''

    select: int = 1
    where: int = 1
    join: int = 2
    group_by: int = 2
    having: int = 3
    subquery: int = 4
    correlated_subquery: int = 5


@dataclass(frozen=True)
class Thresholds:
    '''Score cut-offs mapping a numeric score to a :class:`ComplexityLevel`.

    A score ``s`` is classified as:

    * ``EASY``    if ``s <= easy_max``
    * ``MEDIUM``  if ``easy_max < s <= medium_max``
    * ``HARD``    if ``medium_max < s <= hard_max``
    * ``EXTRA``   otherwise.
    '''

    easy_max: int = 2
    medium_max: int = 5
    hard_max: int = 9


# Aggregation function expression types, mirrored from
# ``constraints/query/aggregation.py``.
_AGG_FUNCS: tuple[type[exp.Expression], ...] = (
    exp.Avg, exp.Count, exp.Sum, exp.Max, exp.Min,
)


@dataclass
class CognitiveBreakdown:
    '''Detailed per-construct contributions to the cognitive score.'''

    select: int = 0
    where: int = 0
    join: int = 0
    group_by: int = 0
    having: int = 0
    subquery: int = 0
    correlated_subquery: int = 0

    @property
    def score(self) -> int:
        '''Total cognitive score (sum of weighted counts).'''
        return (
            self.select
            + self.where
            + self.join
            + self.group_by
            + self.having
            + self.subquery
            + self.correlated_subquery
        )

    def weighted_score(self, weights: Weights) -> int:
        return (
            self.select * weights.select
            + self.where * weights.where
            + self.join * weights.join
            + self.group_by * weights.group_by
            + self.having * weights.having
            + self.subquery * weights.subquery
            + self.correlated_subquery * weights.correlated_subquery
        )

    def as_dict(self) -> dict[str, int]:
        return {
            'select': self.select,
            'where': self.where,
            'join': self.join,
            'group_by': self.group_by,
            'having': self.having,
            'subquery': self.subquery,
            'correlated_subquery': self.correlated_subquery,
        }


def _own_tables(ast: exp.Expression) -> set[str]:
    '''Tables declared in the FROM/JOIN of a single SELECT AST.'''
    own: set[str] = set()
    from_expr = ast.args.get('from_')
    if from_expr is not None and from_expr.this is not None:
        own.add(getattr(from_expr.this, 'real_name', None) or getattr(from_expr.this, 'name', None))
    for join in ast.args.get('joins', []):
        if join.this is not None:
            own.add(getattr(join.this, 'real_name', None) or getattr(join.this, 'name', None))
    own.discard(None)
    return own


def _is_correlated(sub_select: object, parent: object) -> bool:
    '''Heuristic: a subquery is correlated if it references a *column* from a
    table declared in its parent query but not in its own FROM/JOIN.

    We deliberately inspect the subquery's column references rather than
    ``sqlscope``'s ``referenced_tables``, because the latter folds parent
    tables into subqueries (see ``Select._get_referenced_tables``), which would
    make every subquery look correlated.
    '''
    try:
        parent_tables = {t.real_name for t in parent.referenced_tables}  # type: ignore[attr-defined]
        ast = sub_select.ast  # type: ignore[attr-defined]
        if ast is None:
            return False
        own = _own_tables(ast)
        for col in ast.find_all(exp.Column):
            table = col.table
            if table and table not in own and table in parent_tables:
                return True
        return False
    except Exception:
        return False


def cognitive_score(query: Query, *, weights: Weights | None = None) -> tuple[int, CognitiveBreakdown]:
    '''Compute the cognitive complexity score of ``query``.

    Returns the weighted total and a :class:`CognitiveBreakdown` whose fields
    hold the *weighted* per-construct contributions (i.e. count × weight).
    Pass ``weights`` to use a custom weighting (e.g. Giovanna's construct
    table); the default reproduces the ``goals.txt`` table.
    '''
    weights = weights or Weights()

    bd = CognitiveBreakdown()

    for select in query.selects:
        ast = select.ast
        if ast is None:
            continue

        # SELECT — one per SELECT statement.
        bd.select += 1 * weights.select

        # WHERE — one per WHERE clause.
        if select.where is not None:
            bd.where += 1 * weights.where

        # JOIN — one per join edge.
        bd.join += len(list(ast.find_all(exp.Join))) * weights.join

        # GROUP BY — one per GROUP BY clause.
        if len(select.group_by) > 0:
            bd.group_by += 1 * weights.group_by

        # HAVING — one per HAVING clause.
        if select.having is not None:
            bd.having += 1 * weights.having

        # Subqueries — depth information comes from sqlscope.
        for sub, _sql, depth in select.subqueries:
            if depth >= 2:
                continue  # nested subqueries are counted at their own level
            if _is_correlated(sub, select):
                bd.correlated_subquery += 1 * weights.correlated_subquery
            else:
                bd.subquery += 1 * weights.subquery

    return bd.score, bd


def cognitive_hardness(
    score: int,
    *,
    thresholds: Thresholds | None = None,
) -> ComplexityLevel:
    '''Bucket a cognitive score into a :class:`ComplexityLevel`.'''
    thresholds = thresholds or Thresholds()
    if score <= thresholds.easy_max:
        return ComplexityLevel.EASY
    if score <= thresholds.medium_max:
        return ComplexityLevel.MEDIUM
    if score <= thresholds.hard_max:
        return ComplexityLevel.HARD
    return ComplexityLevel.EXTRA


def evaluate(
    query: Query,
    *,
    weights: Weights | None = None,
    thresholds: Thresholds | None = None,
) -> ComplexityLevel:
    '''Score + classify in one call.'''
    score, _ = cognitive_score(query, weights=weights)
    return cognitive_hardness(score, thresholds=thresholds)


def evaluate_sql(
    sql: str,
    *,
    dialect: str | None = None,
    weights: Weights | None = None,
    thresholds: Thresholds | None = None,
) -> ComplexityLevel:
    '''Evaluate the cognitive complexity of a raw SQL string.'''
    parsed = sqlglot.parse_one(sql, dialect=dialect) if dialect else sqlglot.parse_one(sql)
    return evaluate(Query(parsed.sql()), weights=weights, thresholds=thresholds)