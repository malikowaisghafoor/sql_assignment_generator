'''Common types shared by the complexity/hardness metrics (Goal 1).

The metrics implemented in this package all classify a SQL query into one of
four difficulty levels.  ``ComplexityLevel`` is the single enum they map to so
that the different metrics can be compared (see :mod:`complexity.alignment`).
'''

from __future__ import annotations

from enum import Enum


class ComplexityLevel(Enum):
    '''The four difficulty buckets used by every Goal 1 metric.

    The string values match the labels used in the literature ("easy",
    "medium", "hard", "extra hard").
    '''

    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'
    EXTRA = 'extra_hard'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_label(cls, label: str) -> 'ComplexityLevel':
        '''Resolve a level from its (case-insensitive) string label.'''
        normalized = label.strip().lower().replace('-', '_').replace(' ', '_')
        for level in cls:
            if level.value == normalized:
                return level
        raise ValueError(f'Unknown complexity level label: {label!r}')