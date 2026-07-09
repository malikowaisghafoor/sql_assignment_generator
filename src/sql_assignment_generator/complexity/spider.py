'''Spider SQL hardness metric (Goal 1).

Implements the hardness classification published with the Spider benchmark
(Yu et al., 2018) and referenced in ``goals.txt``:

    https://github.com/taoyds/spider/blob/master/evaluation_examples/README.md

The query is first reduced to three counts:

* **components 1** — ``WHERE``, ``GROUP BY``, ``ORDER BY``, ``LIMIT``,
  ``JOIN``, ``OR``, ``LIKE``, ``HAVING``;
* **components 2** — ``EXCEPT``, ``UNION``, ``INTERSECT``, nested
  subqueries;
* **others** — number of aggregations > 1, number of selected columns > 1,
  number of WHERE conditions > 1, number of GROUP BY columns > 1.

which are then combined through Spider's decision tree into one of
:class:`ComplexityLevel`.

All counts are computed from a single ``sqlscope.Query`` by re-using the same
``sqlglot``-based detection patterns already used by the constraint framework,
so the metric stays consistent with the rest of the project.
'''

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlscope import Query

from .base import ComplexityLevel

__all__ = ['SpiderCounts', 'count_spider', 'spider_hardness', 'evaluate', 'evaluate_sql']


# Aggregation function expression types recognised by Spider's "number of
# aggregations" rule.  Mirrors ``constraints/query/aggregation.py``.
_AGG_FUNCS: tuple[type[exp.Expression], ...] = (
    exp.Avg, exp.Count, exp.Sum, exp.Max, exp.Min,
)


@dataclass(frozen=True)
class SpiderCounts:
    '''Raw counts that feed the Spider hardness decision tree.'''

    component1: int
    '''Number of occurrences of [components 1] keywords.'''
    component2: int
    '''Number of occurrences of [components 2] keywords / nested subqueries.'''
    others: int
    '''Number of [Others] rules satisfied (0..4).'''

    # The individual contributions, exposed for transparency/debugging.
    where: int = 0
    group_by: int = 0
    order_by: int = 0
    limit: int = 0
    join: int = 0
    or_: int = 0
    like: int = 0
    having: int = 0
    except_: int = 0
    union: int = 0
    intersect: int = 0
    nested: int = 0
    agg_count: int = 0
    select_columns: int = 0
    where_conditions: int = 0
    group_by_columns: int = 0


def _count_or(where: exp.Expression | None) -> int:
    '''Count ``OR`` connectives inside a WHERE/HAVING expression.'''
    if where is None:
        return 0
    return len(list(where.find_all(exp.Or)))


def _count_like(where: exp.Expression | None) -> int:
    '''Count ``LIKE`` / ``ILIKE`` predicates inside a WHERE/HAVING expression.'''
    if where is None:
        return 0
    return len(list(where.find_all((exp.Like, exp.ILike))))


def _count_where_conditions(where: exp.Expression | None) -> int:
    '''Count leaf conditions in a WHERE clause.

    A flat ``a AND b OR c`` yields three conditions (number of predicates
    joined by AND/OR, plus one).  This matches the convention used by
    ``constraints/query/clause_where.py:Condition``.
    '''
    if where is None:
        return 0
    return 1 + len(list(where.find_all(exp.And))) + len(list(where.find_all(exp.Or)))


def count_spider(query: Query) -> SpiderCounts:
    '''Compute the raw Spider counts for a parsed ``sqlscope.Query``.

    Counts are aggregated across every ``SELECT`` in the query — i.e. the main
    query, every CTE, every subquery and every branch of a set operation —
    because Spider hardness considers the whole query text.
    '''
    where = group_by = order_by = limit = join = or_ = like = having = 0
    except_ = union = intersect_ = nested = 0
    # Totals across selects (reported in SpiderCounts for transparency).
    agg_count = select_columns = where_conditions = group_by_columns = 0
    # Per-select maxima, used for the [Others] rules, which are properties of a
    # single SELECT (e.g. "number of select columns > 1") rather than sums.
    agg_count_max = select_columns_max = where_conditions_max = group_by_columns_max = 0

    # Set operators (UNION / INTERSECT / EXCEPT) sit *above* the individual
    # SELECT branches, so iterating ``query.selects`` (which yields one entry
    # per branch) would miss a top-level set operator.  Parse the full SQL once
    # and count set operators over the whole tree.
    root = sqlglot.parse_one(query.sql)
    union = len(list(root.find_all(exp.Union)))
    intersect_ = len(list(root.find_all(exp.Intersect)))
    except_ = len(list(root.find_all(exp.Except)))

    for select in query.selects:
        ast = select.ast
        if ast is None:
            continue

        # --- components 1 -------------------------------------------------
        if select.where is not None:
            where += 1
        if len(select.group_by) > 0:
            group_by += 1
        if len(select.order_by) > 0:
            order_by += 1
        if select.limit is not None:
            limit += 1

        # JOINs: number of join clauses (matches Spider counting joins as
        # table_units - 1, i.e. one per join edge).
        join += len(list(ast.find_all(exp.Join)))

        # OR / LIKE inside WHERE and HAVING.
        or_ += _count_or(select.where) + _count_or(select.having)
        like += _count_like(select.where) + _count_like(select.having)

        if select.having is not None:
            having += 1

        # --- components 2 (nested subqueries) ----------------------------
        # (Set operators UNION/INTERSECT/EXCEPT are counted on the root tree
        # above, since they sit above the SELECT branches.)

        # Nested subqueries: count subqueries whose nesting depth >= 2.
        for _sub, _sql, depth in select.subqueries:
            if depth >= 2:
                nested += 1

        # --- others (per-select) -----------------------------------------
        # Spider's [Others] rules are properties of a single SELECT (e.g.
        # "number of select columns > 1"), so we track the maximum across
        # branches rather than summing across them.
        sel_agg = len(list(ast.find_all(_AGG_FUNCS)))
        sel_cols = len(select.output.columns)
        sel_wheres = _count_where_conditions(select.where)
        sel_gbs = len(select.group_by)
        agg_count += sel_agg
        select_columns += sel_cols
        where_conditions += sel_wheres
        group_by_columns += sel_gbs
        agg_count_max = max(agg_count_max, sel_agg)
        select_columns_max = max(select_columns_max, sel_cols)
        where_conditions_max = max(where_conditions_max, sel_wheres)
        group_by_columns_max = max(group_by_columns_max, sel_gbs)

    component1 = where + group_by + order_by + limit + join + or_ + like + having
    component2 = except_ + union + intersect_ + nested

    others = 0
    if agg_count_max > 1:
        others += 1
    if select_columns_max > 1:
        others += 1
    if where_conditions_max > 1:
        others += 1
    if group_by_columns_max > 1:
        others += 1

    return SpiderCounts(
        component1=component1,
        component2=component2,
        others=others,
        where=where,
        group_by=group_by,
        order_by=order_by,
        limit=limit,
        join=join,
        or_=or_,
        like=like,
        having=having,
        except_=except_,
        union=union,
        intersect=intersect_,
        nested=nested,
        agg_count=agg_count,
        select_columns=select_columns,
        where_conditions=where_conditions,
        group_by_columns=group_by_columns,
    )


def spider_hardness(counts: SpiderCounts) -> ComplexityLevel:
    '''Apply the Spider decision tree to pre-computed counts.

    The rules are transcribed verbatim from ``goals.txt``:

    * **Easy** — zero or exactly one keyword from [components 1], no rule from
      [Others] satisfied, and no keyword from [components 2].
    * **Medium** — (no more than two rules from [Others], no more than one
      keyword from [components 1], nothing from [components 2]) *or* (exactly
      two keywords from [components 1] and fewer than two rules from [Others],
      nothing from [components 2]).
    * **Hard** — (more than two rules from [Others], at most two keywords from
      [components 1], nothing from [components 2]) *or* (two < keywords from
      [components 1] <= 3 and at most two rules from [Others], nothing from
      [components 2]) *or* (at most one keyword from [components 1], no rule
      from [Others], and exactly one keyword from [components 2]).
    * **Extra Hard** — everything else.
    '''
    c1, c2, o = counts.component1, counts.component2, counts.others

    # Easy
    if c1 <= 1 and o == 0 and c2 == 0:
        return ComplexityLevel.EASY

    # Medium
    medium_a = (o <= 2 and c1 <= 1 and c2 == 0)
    medium_b = (c1 == 2 and o < 2 and c2 == 0)
    if medium_a or medium_b:
        return ComplexityLevel.MEDIUM

    # Hard
    hard_a = (o > 2 and c1 <= 2 and c2 == 0)
    hard_b = (2 < c1 <= 3 and o <= 2 and c2 == 0)
    hard_c = (c1 <= 1 and o == 0 and c2 == 1)
    if hard_a or hard_b or hard_c:
        return ComplexityLevel.HARD

    # Everything else
    return ComplexityLevel.EXTRA


def evaluate(query: Query) -> ComplexityLevel:
    '''Convenience: count + classify in one call.'''
    return spider_hardness(count_spider(query))


def evaluate_sql(sql: str, *, dialect: str | None = None) -> ComplexityLevel:
    '''Evaluate the Spider hardness of a raw SQL string.'''
    parsed = sqlglot.parse_one(sql, dialect=dialect) if dialect else sqlglot.parse_one(sql)
    query = Query(parsed.sql())
    return evaluate(query)