"""Optimal-strategy baselines for matching-pairs (used as efficiency reference).

Each module here exports a `compute_*` function that takes the raw `board.cards`
2D list and returns the resp_times an idealized player would need.
"""

from common.optimal.greedy_scan import compute_optimal_resp_times

__all__ = ["compute_optimal_resp_times"]
