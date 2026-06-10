"""Matrix conversion helpers."""

from __future__ import annotations

import numpy as np


def pfm_to_pwm(pfm):
    """Convert Position Frequency Matrix to Position Weight Matrix."""
    return np.log((pfm + 0.0001) / 0.25)


def pcm_to_pfm(pcm, pseudocount: float = 0.25):
    """Convert Position Count Matrix to Position Frequency Matrix."""
    number_of_sites = pcm.sum(axis=0)
    nuc_pseudo = float(pseudocount)
    return (pcm + nuc_pseudo) / (number_of_sites + 4.0 * nuc_pseudo)
