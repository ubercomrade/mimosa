"""ROC and precision-recall curve helpers."""

from __future__ import annotations

import numpy as np


def precision_recall_curve(classification, scores):
    """Compute the precision-recall curve."""
    classification = np.asarray(classification)
    scores = np.asarray(scores)

    if len(scores) == 0:
        return np.array([1.0]), np.array([0.0]), np.array([np.inf])

    indexes = np.argsort(scores)[::-1]
    sorted_scores = scores[indexes]
    sorted_classification = classification[indexes]
    number_of_uniq_scores = np.unique(scores).shape[0]
    max_size = number_of_uniq_scores + 1

    precision = np.zeros(max_size)
    recall = np.zeros(max_size)
    uniq_scores = np.zeros(max_size)
    precision[0] = 1.0
    recall[0] = 0.0
    uniq_scores[0] = np.inf

    true_positive = 0
    false_positive = 0
    number_of_true = np.sum(classification == 1)
    number_of_false = np.sum(classification == 0)
    true_false_ratio = 1.0 if number_of_false == 0 else number_of_true / number_of_false

    position = 1
    current_score = sorted_scores[0]

    for index, score in enumerate(sorted_scores):
        flag = sorted_classification[index]
        if flag == 1:
            true_positive += 1
        else:
            false_positive += 1

        if index == len(scores) - 1 or current_score != sorted_scores[index + 1]:
            uniq_scores[position] = score
            denominator = true_positive + true_false_ratio * false_positive
            precision[position] = true_positive / denominator if denominator > 0 else 1.0
            recall[position] = true_positive / number_of_true if number_of_true > 0 else 0.0
            position += 1
            if index < len(scores) - 1:
                current_score = sorted_scores[index + 1]

    return precision[:position], recall[:position], uniq_scores[:position]


def roc_curve(classification, scores):
    """Compute the ROC curve."""
    classification = np.asarray(classification)
    scores = np.asarray(scores)

    if len(scores) == 0:
        return np.array([0.0]), np.array([0.0]), np.array([np.inf])

    indexes = np.argsort(scores)[::-1]
    sorted_scores = scores[indexes]
    sorted_classification = classification[indexes]
    number_of_uniq_scores = np.unique(scores).shape[0]
    max_size = number_of_uniq_scores + 1

    tpr = np.zeros(max_size)
    fpr = np.zeros(max_size)
    uniq_scores = np.zeros(max_size)
    uniq_scores[0] = np.inf

    true_positive = 0
    false_positive = 0
    number_of_true = np.sum(classification == 1)
    number_of_false = np.sum(classification == 0)
    position = 1
    current_score = sorted_scores[0]

    for index, score in enumerate(sorted_scores):
        flag = sorted_classification[index]
        if flag == 1:
            true_positive += 1
        else:
            false_positive += 1

        if index == len(scores) - 1 or current_score != sorted_scores[index + 1]:
            uniq_scores[position] = score
            tpr[position] = true_positive / number_of_true if number_of_true > 0 else 0.0
            fpr[position] = false_positive / number_of_false if number_of_false > 0 else 0.0
            position += 1
            if index < len(scores) - 1:
                current_score = sorted_scores[index + 1]

    return tpr[:position], fpr[:position], uniq_scores[:position]


def cut_roc(tpr: np.ndarray, fpr: np.ndarray, thr: np.ndarray, score_cutoff: float):
    """Truncate ROC curve at a specific score threshold."""
    if score_cutoff == -np.inf:
        return tpr, fpr, thr

    mask = thr >= score_cutoff
    if not np.any(mask):
        return (
            np.array([tpr[0]], dtype=tpr.dtype),
            np.array([0.0], dtype=fpr.dtype),
            np.array([score_cutoff], dtype=thr.dtype),
        )

    last = int(np.where(mask)[0][-1])
    if thr[last] == score_cutoff or last == len(thr) - 1:
        return tpr[: last + 1], fpr[: last + 1], thr[: last + 1]

    s0, s1 = float(thr[last]), float(thr[last + 1])
    t0, t1 = float(tpr[last]), float(tpr[last + 1])
    f0, f1 = float(fpr[last]), float(fpr[last + 1])
    alpha = 0.0 if s0 == s1 else (score_cutoff - s0) / (s1 - s0)
    t_cut = t0 + alpha * (t1 - t0)
    f_cut = f0 + alpha * (f1 - f0)

    tpr_cut = np.concatenate([tpr[: last + 1], np.array([t_cut], dtype=tpr.dtype)])
    fpr_cut = np.concatenate([fpr[: last + 1], np.array([f_cut], dtype=fpr.dtype)])
    thr_cut = np.concatenate([thr[: last + 1], np.array([score_cutoff], dtype=thr.dtype)])
    return tpr_cut, fpr_cut, thr_cut


def cut_prc(rec: np.ndarray, prec: np.ndarray, thr: np.ndarray, score_cutoff: float):
    """Truncate Precision-Recall curve at a specific score threshold."""
    if score_cutoff == -np.inf:
        return rec, prec, thr

    mask = thr >= score_cutoff
    if not np.any(mask):
        return (
            np.array([0.0], dtype=rec.dtype),
            np.array([prec[0]], dtype=prec.dtype),
            np.array([score_cutoff], dtype=thr.dtype),
        )

    last = int(np.where(mask)[0][-1])
    if thr[last] == score_cutoff or last == len(thr) - 1:
        return rec[: last + 1], prec[: last + 1], thr[: last + 1]

    s0, s1 = float(thr[last]), float(thr[last + 1])
    r0, r1 = float(rec[last]), float(rec[last + 1])
    p0, p1 = float(prec[last]), float(prec[last + 1])
    alpha = 0.0 if s0 == s1 else (score_cutoff - s0) / (s1 - s0)
    r_cut = r0 + alpha * (r1 - r0)
    p_cut = p0 + alpha * (p1 - p0)

    rec_cut = np.concatenate([rec[: last + 1], np.array([r_cut], dtype=rec.dtype)])
    prec_cut = np.concatenate([prec[: last + 1], np.array([p_cut], dtype=prec.dtype)])
    thr_cut = np.concatenate([thr[: last + 1], np.array([score_cutoff], dtype=thr.dtype)])
    return rec_cut, prec_cut, thr_cut


def standardized_pauc(pauc_raw: float, pauc_min: float, pauc_max: float) -> float:
    """Standardize a partial AUC value to the range [0.5, 1.0]."""
    denom = pauc_max - pauc_min
    if denom <= 0:
        return 0.5
    return 0.5 * (1.0 + (pauc_raw - pauc_min) / denom)
