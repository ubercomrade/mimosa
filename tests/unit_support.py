# ruff: noqa: F401
"""
Unit tests for key computational functions in mimosa.

These tests validate the correctness of individual functions from:
- mimosa/functions.py
- mimosa/comparison.py
- mimosa/models.py
"""

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from scipy import stats

import mimosa
import mimosa.api as api_module
from mimosa.api import (
    compare_one_to_many,
    compare_one_to_one,
    create_null_distribution,
    create_null_distribution_config,
    create_one_to_many_config,
    create_one_to_one_config,
    run_null_distribution,
    run_one_to_many,
    run_one_to_one,
)
from mimosa.batches import (
    flatten_profile_bundle,
    flatten_valid,
    make_score_batch,
    make_sequence_batch,
    make_strand_bundle,
    profile_row_values,
    row_values,
)
from mimosa.cache import clear_cache
from mimosa.cli import build_null_request_from_args, map_args_to_comparator_kwargs
from mimosa.comparison import (
    compare,
    create_comparator_config,
    strategy_motif,
    strategy_profile,
)
from mimosa.comparison import registry as comparison_registry
from mimosa.functions import (
    apply_score_log_tail_table,
    batch_all_scores,
    batch_all_scores_strands,
    build_score_log_tail_table,
    calc_co,
    calc_dice,
    cut_prc,
    cut_roc,
    format_params,
    normalize_empirical_log_tail_pair,
    pcm_to_pfm,
    pfm_to_pwm,
    precision_recall_curve,
    roc_curve,
    rowwise_co,
    rowwise_cosine,
    rowwise_dice,
    score_seq,
    scores_to_empirical_log_tail,
    standardized_pauc,
)
from mimosa.io import (
    parse_file_content,
    read_meme,
    read_meme_many,
    read_pfm,
    read_scores,
    read_sitega,
    read_slim,
    write_dist,
)
from mimosa.models import GenericModel, ModelHandler, read_model, read_models, write_model
from mimosa.models import registry as model_registry
from mimosa.nulls import (
    GenextremeSurvivalEstimator,
    NullBuildRequest,
    NullBuildSummary,
    adjusted_pvalues,
    annotate_results_with_nulls,
    build_null_distributions,
    environment_metadata,
    fit_survival_estimator,
    is_null_distribution_file_compatible,
    parse_group_relations,
    run_build_null_request,
    save_null_distribution_file,
)
from mimosa.scanning import calculate_threshold_table, get_frequencies, scan_model, scan_model_strands
from mimosa.sites import get_pfm, get_sites
from mimosa.types import ComparisonResult

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "models"
EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"
PLUS_STRAND = 0
MINUS_STRAND = 1
_DNA_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
_RC_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
_JSTACS_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")
_LOG_UNIFORM_BASE = float(np.log(4.0))


def _encode_sequence(sequence: str) -> np.ndarray:
    """Encode an ACGT string as the project's integer alphabet."""
    return np.array([_DNA_TO_INT[symbol] for symbol in sequence], dtype=np.int8)


def _reverse_complement_encoded(sequence: np.ndarray) -> np.ndarray:
    """Return the reverse complement in the project's integer alphabet."""
    return _RC_TABLE[sequence[::-1]]


def _score_batch_from_flat(data: np.ndarray, offsets: np.ndarray):
    """Build one score batch from flattened values and ragged offsets."""
    rows = [np.asarray(data[offsets[index] : offsets[index + 1]]) for index in range(len(offsets) - 1)]
    return make_score_batch(rows)


def _make_scores_model(name: str, rows: list[list[float]] | list[np.ndarray]) -> GenericModel:
    """Build one GenericModel backed by precomputed score rows."""
    batch = make_score_batch([np.asarray(row, dtype=np.float32) for row in rows])
    return GenericModel(type_key="scores", name=name, representation=None, length=0, config={"scores_data": batch})


def _make_shifted_core_pwm_model(
    name: str,
    core_offset: int,
    core: tuple[int, ...] = (0, 1, 2),
    motif_length: int = 7,
) -> GenericModel:
    """Build one PWM with an informative core placed at a requested matrix offset."""
    pfm = np.full((4, motif_length), 0.25, dtype=np.float32)
    for column_delta, base_index in enumerate(core):
        column_index = core_offset + column_delta
        pfm[:, column_index] = 0.001
        pfm[base_index, column_index] = 0.997
        pfm[:, column_index] /= pfm[:, column_index].sum()

    pwm = pfm_to_pwm(pfm)
    representation = np.concatenate((pwm, np.min(pwm, axis=0, keepdims=True)), axis=0).astype(np.float32)
    return GenericModel("pwm", name, representation, motif_length, {"kmer": 1, "_source_pfm": pfm})


def _xml_numeric_value_reference(elem: ET.Element | None) -> float | None:
    """Extract the last numeric scalar from a Jstacs XML element."""
    if elem is None:
        return None

    texts = [text.strip() for text in elem.itertext() if text and text.strip()]
    for text in reversed(texts):
        if _JSTACS_NUMERIC_RE.fullmatch(text):
            return float(text)

    return None


def _xml_array_reference(elem: ET.Element):
    """Recursively convert Jstacs <pos>-based arrays to Python lists."""
    pos_children = [child for child in elem if child.tag == "pos"]
    if not pos_children:
        return _xml_numeric_value_reference(elem)

    return [_xml_array_reference(child) for child in pos_children]


def _logsumexp_reference(values: np.ndarray) -> float:
    """Compute log-sum-exp in float64 for Java-reference scoring."""
    shifted = values - np.max(values)
    return float(np.max(values) + np.log(np.sum(np.exp(shifted))))


def _parse_dimont_tree_reference(elem: ET.Element) -> dict:
    """Parse one Dimont parameter tree using only XML state."""
    pars = elem.find("pars")
    pars_pos = [child for child in pars if child.tag == "pos"] if pars is not None else []
    if pars_pos:
        values = np.full(4, -np.inf, dtype=np.float64)
        for pos in pars_pos:
            parameter = pos.find("parameter")
            symbol = int(_xml_numeric_value_reference(parameter.find("symbol")))
            values[symbol] = float(_xml_numeric_value_reference(parameter.find("value")))
        return {"scores": values}

    children = elem.find("children")
    assert children is not None
    child_nodes = [None] * 4
    for pos in children:
        if pos.tag == "pos":
            child_nodes[int(pos.attrib["val"])] = _parse_dimont_tree_reference(pos.find("treeElement"))

    return {
        "context_pos": int(_xml_numeric_value_reference(elem.find("contextPos"))),
        "children": child_nodes,
    }


@lru_cache(maxsize=None)
def _load_dimont_reference(path: str) -> tuple[dict, ...]:
    """Load the Java getLogScoreFor-equivalent Dimont tree structure from XML."""
    root = ET.parse(path).getroot()
    model = root.find(".//ThresholdedStrandChIPper/function/pos/MarkovModelDiffSM")
    assert model is not None
    trees = model.find("bayesianNetworkSF/trees")
    assert trees is not None
    parsed_trees = []
    for pos in trees:
        if pos.tag != "pos":
            continue
        node = _parse_dimont_tree_reference(pos.find("parameterTree/root/treeElement"))
        parsed_trees.append(node)

    return tuple(parsed_trees)


def _reference_dimont_site_score(path: Path, sequence: np.ndarray) -> float:
    """Evaluate one site with Dimont raw score plus uniform-background log-odds correction."""
    trees = _load_dimont_reference(str(path))
    total = 0.0
    for position, tree in enumerate(trees):
        node = tree
        while "scores" not in node:
            node = node["children"][int(sequence[node["context_pos"]])]
        total += float(node["scores"][int(sequence[position])])
    return total + len(sequence) * _LOG_UNIFORM_BASE


@lru_cache(maxsize=None)
def _load_slim_reference(path: str) -> tuple[list, list, list]:
    """Load the raw Java SLIM parameter arrays from XML."""
    root = ET.parse(path).getroot()
    slim = root.find(".//SLIM")
    assert slim is not None
    component = _xml_array_reference(slim.find("componentMixtureParameters"))
    ancestor = _xml_array_reference(slim.find("ancestorMixtureParameters"))
    dependency = _xml_array_reference(slim.find("dependencyParameters"))
    return component, ancestor, dependency


def _reference_slim_site_score(path: Path, sequence: np.ndarray) -> float:
    """Evaluate one site as Slim log-odds against a uniform single-base background."""
    component, ancestor, dependency = _load_slim_reference(str(path))
    score = 0.0
    alphabet_size = 4

    def get_offset(start: int, order: int) -> int:
        offset = 0
        current_order = 1
        while current_order < order:
            offset = offset * alphabet_size + int(sequence[start - current_order])
            current_order += 1
        return offset

    def next_context(context: int, position: int, component_index: int, ancestor_index: int) -> int:
        width = len(dependency[position][component_index][0])
        return (context * width + int(sequence[position - component_index - ancestor_index])) % len(
            dependency[position][component_index]
        )

    for position in range(len(component)):
        current_nt = int(sequence[position])
        component_logits = np.asarray(component[position], dtype=np.float64)
        component_log_norm = _logsumexp_reference(component_logits)
        local_scores = []

        independent_logits = np.asarray(dependency[position][0][0], dtype=np.float64)
        independent_log_norm = _logsumexp_reference(independent_logits)
        local_scores.append(
            component_logits[0] - component_log_norm + independent_logits[current_nt] - independent_log_norm
        )

        for component_index in range(1, len(component[position])):
            ancestor_logits = np.asarray(ancestor[position][component_index], dtype=np.float64)
            ancestor_log_norm = _logsumexp_reference(ancestor_logits)
            context = get_offset(position, component_index)
            ancestor_scores = []

            for ancestor_index in range(len(ancestor[position][component_index])):
                context = next_context(context, position, component_index, ancestor_index)
                dependency_logits = np.asarray(dependency[position][component_index][context], dtype=np.float64)
                dependency_log_norm = _logsumexp_reference(dependency_logits)
                ancestor_scores.append(
                    ancestor_logits[ancestor_index]
                    - ancestor_log_norm
                    + dependency_logits[current_nt]
                    - dependency_log_norm
                )

            local_scores.append(
                component_logits[component_index]
                - component_log_norm
                + _logsumexp_reference(np.asarray(ancestor_scores))
            )

        score += _logsumexp_reference(np.asarray(local_scores))

    return score + len(sequence) * _LOG_UNIFORM_BASE


__all__ = [name for name in globals() if not name.startswith("__")]
