'''Goal 1 — SQL query complexity / hardness metrics.

This package implements the four assessment directions listed under
``Possible Task`` in ``goals.txt``:

* :mod:`complexity.spider`      — Spider benchmark hardness (Yu et al., 2018).
* :mod:`complexity.cognitive`   — construct-weighted "cognitive" complexity
  (supports custom per-construct weights, e.g. Giovanna's table).
* :mod:`complexity.structural`  — the "currently used" weighted structural
  score (number of tables, joins, subqueries, nesting depth, ...).
* :mod:`complexity.alignment`   — helpers to run all metrics side-by-side and
  check whether they are aligned.

Empirical difficulty (inferred from student performance) is intentionally out
of scope for this code milestone: it requires interaction data and is left as
future work documented in ``COMPLEXITY.md``.

Every metric maps a ``sqlscope.Query`` to the shared
:class:`ComplexityLevel` enum, so different metrics are directly comparable.
'''

from .base import ComplexityLevel
from . import spider, cognitive, structural, alignment

__all__ = [
    'ComplexityLevel',
    'spider',
    'cognitive',
    'structural',
    'alignment',
]