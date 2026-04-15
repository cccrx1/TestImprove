"""Baseline defense registry for REFINE-centered workflows.

This module keeps the remaining non-REFINE defenses discoverable without
mixing them into the main thesis workflow.
"""

from .ABL import ABL
from .NAD import NAD
from .Pruning import Pruning


BASELINE_DEFENSES = {
    'abl': ABL,
    'nad': NAD,
    'pruning': Pruning,
}


def get_baseline_defense(name):
    key = name.lower()
    if key not in BASELINE_DEFENSES:
        supported = ', '.join(sorted(BASELINE_DEFENSES))
        raise KeyError(f'Unknown baseline defense: {name}. Supported: {supported}.')
    return BASELINE_DEFENSES[key]


def list_baseline_defenses():
    return sorted(BASELINE_DEFENSES)
