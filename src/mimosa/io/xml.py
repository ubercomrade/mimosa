"""Jstacs XML model readers for Dimont and Slim motif models."""

from __future__ import annotations

import itertools
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np

_JSTACS_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")
_LOG_UNIFORM_BASE = float(np.log(4.0))
XmlArray = float | list["XmlArray"]


def _xml_numeric_value(elem: Optional[ET.Element]) -> Optional[float]:
    """Extract the last numeric scalar from a Jstacs XML element."""
    if elem is None:
        return None

    texts = [text.strip() for text in elem.itertext() if text and text.strip()]
    for text in reversed(texts):
        if _JSTACS_NUMERIC_RE.fullmatch(text):
            return float(text)

    return None


def _required_xml_numeric(elem: ET.Element | None, label: str) -> float:
    """Extract one required numeric XML scalar with a domain-specific error."""
    value = _xml_numeric_value(elem)
    if value is None:
        raise ValueError(f"Malformed XML: missing numeric value for {label}")
    return value


def _required_xml_child(parent: ET.Element, path: str, label: str) -> ET.Element:
    """Return one required XML child with a domain-specific error."""
    element = parent.find(path)
    if element is None:
        raise ValueError(f"Malformed XML: missing {label}")
    return element


def _xml_array(elem: ET.Element) -> XmlArray | None:
    """Recursively convert Jstacs <pos>-based arrays to Python lists."""
    pos_children = [child for child in elem if child.tag == "pos"]
    if not pos_children:
        return _xml_numeric_value(elem)

    values = []
    for child in pos_children:
        value = _xml_array(child)
        if value is None:
            raise ValueError("Malformed XML: empty XML array position")
        values.append(value)
    return values


def _required_xml_array(elem: ET.Element | None, label: str) -> XmlArray:
    """Convert one required Jstacs array element to nested Python lists."""
    if elem is None:
        raise ValueError(f"Malformed XML: missing {label}")
    value = _xml_array(elem)
    if value is None:
        raise ValueError(f"Malformed XML: empty {label}")
    return value


def _log_normalize(values: np.ndarray) -> np.ndarray:
    """Convert unconstrained log-parameters to normalized log-probabilities."""
    return values - _logsumexp(values)


def _logsumexp(values: np.ndarray) -> float:
    """Compute a stable float64 log-sum-exp."""
    shifted = values - np.max(values)
    return float(np.max(values) + np.log(np.sum(np.exp(shifted))))


def _fill_n_axis_with_min(arr: np.ndarray, axis: int) -> None:
    """Assign the N state on one axis to the minimum over concrete nucleotides."""
    index: list[int | slice] = [slice(None)] * arr.ndim
    index[axis] = 4
    arr[tuple(index)] = np.min(np.take(arr, [0, 1, 2, 3], axis=axis), axis=axis)


def _build_position_tensor(
    context_log_probs: Dict[Tuple[int, ...], np.ndarray],
    context_axes: List[int],
    span: int,
) -> np.ndarray:
    """Build one dense 5-ary context tensor for a single motif position."""
    temp = np.full((5,) * span + (4,), np.inf, dtype=np.float64)

    for context_values, log_probs in context_log_probs.items():
        assignment = {axis: value for axis, value in zip(context_axes, context_values, strict=False)}
        index: list[int | slice] = []
        for axis in range(span):
            index.append(assignment.get(axis, slice(None)))
        index.append(slice(None))
        temp[tuple(index)] = log_probs

    for axis in context_axes:
        _fill_n_axis_with_min(temp, axis)

    position_tensor = np.empty((5,) * (span + 1), dtype=np.float64)
    position_tensor[..., :4] = temp
    position_tensor[..., 4] = np.min(temp, axis=-1)
    return position_tensor.astype(np.float32, copy=False)


def _context_value(full_context: Tuple[int, ...], span: int, position: int, absolute_position: int) -> int:
    """Map one absolute parent position to the corresponding dense context axis."""
    if absolute_position < 0:
        raise ValueError(f"Model references position {absolute_position} before the motif start at position {position}")

    axis = absolute_position - (position - span)
    if axis < 0 or axis >= span:
        raise ValueError(f"Context position {absolute_position} at motif position {position} does not fit span {span}")

    return full_context[axis]


def _iter_full_contexts(span: int) -> Iterable[Tuple[int, ...]]:
    """Iterate over all concrete A/C/G/T contexts of the requested span."""
    return itertools.product(range(4), repeat=span)


def _find_xml_element(root: ET.Element, xpath: str, error_message: str) -> ET.Element:
    """Return one required XML element or raise a descriptive error."""
    element = root.find(xpath)
    if element is None:
        raise ValueError(error_message)
    return element


def _parse_slim_model(path: str) -> tuple[int, list, list, list]:
    """Parse raw SLIM arrays from XML."""
    root = ET.parse(path).getroot()
    slim = _find_xml_element(root, ".//SLIM", f"Could not find SLIM model in {path}")
    length = int(_required_xml_numeric(slim.find("length"), "SLIM length"))
    _distance = int(_required_xml_numeric(slim.find("distance"), "SLIM distance"))
    component_params = cast(
        list,
        _required_xml_array(slim.find("componentMixtureParameters"), "SLIM componentMixtureParameters"),
    )
    ancestor_params = cast(
        list,
        _required_xml_array(slim.find("ancestorMixtureParameters"), "SLIM ancestorMixtureParameters"),
    )
    dependency_params = cast(
        list,
        _required_xml_array(slim.find("dependencyParameters"), "SLIM dependencyParameters"),
    )
    return length, component_params, ancestor_params, dependency_params


def _slim_span(length: int, component_params: list, ancestor_params: list, path: str) -> int:
    """Compute the dense context span implied by a SLIM model."""
    span = 0
    for position in range(length):
        for component_index in range(1, len(component_params[position])):
            ancestor_count = len(ancestor_params[position][component_index])
            if ancestor_count <= 0:
                raise ValueError(f"Malformed SLIM model in {path}: empty ancestor mixture at position {position}")
            span = max(span, component_index + ancestor_count - 1)
    return span


def _normalize_slim_parameters(path: str) -> dict:
    """Normalize raw SLIM parameters to log-probability tables."""
    length, component_params, ancestor_params, dependency_params = _parse_slim_model(path)
    return {
        "length": length,
        "span": _slim_span(length, component_params, ancestor_params, path),
        "alphabet_size": len(dependency_params[0][0][0]),
        "component_log_probs": [
            _log_normalize(np.asarray(component_params[position], dtype=np.float64)) for position in range(length)
        ],
        "ancestor_log_probs": [
            [_log_normalize(np.asarray(component, dtype=np.float64)) for component in ancestor_params[position]]
            for position in range(length)
        ],
        "dependency_log_probs": [
            [
                np.vstack(
                    [
                        _log_normalize(np.asarray(row, dtype=np.float64))
                        for row in dependency_params[position][component]
                    ]
                )
                for component in range(len(dependency_params[position]))
            ]
            for position in range(length)
        ],
    }


def _slim_symbol_log_probs(
    position: int,
    symbol: int,
    full_context: Tuple[int, ...],
    params: dict,
) -> float:
    """Evaluate one SLIM position for a concrete symbol and context."""
    component_log_probs = params["component_log_probs"]
    dependency_log_probs = params["dependency_log_probs"]
    ancestor_log_probs = params["ancestor_log_probs"]
    alphabet_size = int(params["alphabet_size"])

    local_scores = np.empty(len(component_log_probs[position]), dtype=np.float64)
    local_scores[0] = component_log_probs[position][0] + dependency_log_probs[position][0][0, symbol]

    for component_index in range(1, len(component_log_probs[position])):
        ancestor_count = len(ancestor_log_probs[position][component_index])
        context_index = 0

        for current_order in range(1, component_index):
            parent_position = position - current_order
            context_index = context_index * alphabet_size + _context_value(
                full_context, int(params["span"]), position, parent_position
            )

        ancestor_scores = np.empty(ancestor_count, dtype=np.float64)
        dependency = dependency_log_probs[position][component_index]
        total_contexts = dependency.shape[0]
        width = dependency.shape[1]

        for ancestor_index in range(ancestor_count):
            parent_position = position - component_index - ancestor_index
            context_index = (
                context_index * width + _context_value(full_context, int(params["span"]), position, parent_position)
            ) % total_contexts
            ancestor_scores[ancestor_index] = (
                ancestor_log_probs[position][component_index][ancestor_index] + dependency[context_index, symbol]
            )

        local_scores[component_index] = component_log_probs[position][component_index] + _logsumexp(ancestor_scores)

    return _logsumexp(local_scores) + _LOG_UNIFORM_BASE


def _build_slim_position_tensor(position: int, params: dict, full_contexts: list[Tuple[int, ...]]) -> np.ndarray:
    """Materialize one SLIM position into a dense tensor."""
    context_log_probs = {}
    for full_context in full_contexts:
        symbol_log_probs = np.empty(4, dtype=np.float64)
        for symbol in range(4):
            symbol_log_probs[symbol] = _slim_symbol_log_probs(position, symbol, full_context, params)
        context_log_probs[full_context] = symbol_log_probs
    span = int(params["span"])
    return _build_position_tensor(context_log_probs, list(range(span)), span)


def read_slim(path: str) -> tuple[np.ndarray, int, int]:
    """Read a Jstacs Slim XML model into a dense log-odds tensor."""
    params = _normalize_slim_parameters(path)
    span = int(params["span"])
    length = int(params["length"])
    tensor = np.empty((5,) * (span + 1) + (length,), dtype=np.float32)
    full_contexts = list(_iter_full_contexts(span))
    for position in range(length):
        tensor[..., position] = _build_slim_position_tensor(position, params, full_contexts)
    return tensor, length, span


def _parse_dimont_treeelement(elem: ET.Element) -> dict:
    """Parse one recursive MarkovModelDiffSM tree element."""
    node: dict[str, Any] = {
        "context_pos": int(_required_xml_numeric(elem.find("contextPos"), "Dimont tree contextPos"))
    }

    pars = elem.find("pars")
    pars_pos = [child for child in pars if child.tag == "pos"] if pars is not None else []

    if pars_pos:
        scores = np.full(4, -np.inf, dtype=np.float64)

        for pos in pars_pos:
            par = _required_xml_child(pos, "parameter", "Dimont tree parameter")
            symbol = int(_required_xml_numeric(par.find("symbol"), "Dimont tree parameter symbol"))
            scores[symbol] = _required_xml_numeric(par.find("value"), "Dimont tree parameter value")

        node["scores"] = scores
        return node

    children_elem = elem.find("children")
    if children_elem is None:
        raise ValueError("Malformed Dimont tree: expected children or parameters")

    children: list[dict[str, Any] | None] = [None] * 4
    for pos in children_elem:
        if pos.tag != "pos":
            continue
        child_index = int(pos.attrib["val"])
        children[child_index] = _parse_dimont_treeelement(
            _required_xml_child(pos, "treeElement", "Dimont child treeElement")
        )

    if any(child is None for child in children):
        raise ValueError("Malformed Dimont tree: expected 4 children")

    node["children"] = children
    return node


def _parse_dimont_model(path: str) -> tuple[list[list[int]], list[dict]]:
    """Parse Dimont context positions and parameter trees from XML."""
    root = ET.parse(path).getroot()
    model = _find_xml_element(
        root,
        ".//ThresholdedStrandChIPper/function/pos/MarkovModelDiffSM",
        f"Could not find Dimont motif model in {path}",
    )
    trees = _find_xml_element(model, "bayesianNetworkSF/trees", f"Malformed Dimont model in {path}: missing trees")
    context_positions: List[List[int]] = []
    nodes = []
    for pos in trees:
        if pos.tag != "pos":
            continue
        parameter_tree = _find_xml_element(
            pos, "parameterTree", f"Malformed Dimont model in {path}: missing parameter tree"
        )
        context_pos_elem = parameter_tree.find("contextPoss")
        current_context_positions = (
            [
                int(_required_xml_numeric(child, "Dimont contextPoss position"))
                for child in context_pos_elem
                if child.tag == "pos"
            ]
            if context_pos_elem is not None
            else []
        )
        context_positions.append(current_context_positions)
        nodes.append(
            _parse_dimont_treeelement(
                _required_xml_child(parameter_tree, "root/treeElement", "Dimont root treeElement")
            )
        )
    return context_positions, nodes


def _dimont_span(context_positions: list[list[int]]) -> int:
    """Compute the dense context span implied by Dimont parent links."""
    span = 0
    for position, positions in enumerate(context_positions):
        if positions:
            span = max(span, max(position - parent for parent in positions))
    return span


def _normalize_dimont_parameters(path: str) -> dict:
    """Normalize Dimont XML state to a tensor-materialization plan."""
    context_positions, nodes = _parse_dimont_model(path)
    return {
        "length": len(nodes),
        "span": _dimont_span(context_positions),
        "nodes": nodes,
    }


def _build_dimont_position_tensor(
    position: int,
    node: dict,
    span: int,
    full_contexts: list[Tuple[int, ...]],
) -> np.ndarray:
    """Materialize one Dimont position into a dense site-scoring tensor."""
    context_scores = {}

    for full_context in full_contexts:
        current = node

        while "scores" not in current:
            parent_nt = _context_value(
                full_context,
                span,
                position,
                current["context_pos"],
            )
            current = current["children"][parent_nt]

        context_scores[full_context] = current["scores"] + _LOG_UNIFORM_BASE

    return _build_position_tensor(context_scores, list(range(span)), span)


def read_dimont(path: str) -> tuple[np.ndarray, int, int]:
    """Read a Jstacs Dimont XML model into a dense site-scoring tensor."""
    params = _normalize_dimont_parameters(path)
    span = int(params["span"])
    length = int(params["length"])

    tensor = np.empty((5,) * (span + 1) + (length,), dtype=np.float32)
    full_contexts = list(_iter_full_contexts(span))

    for position, node in enumerate(params["nodes"]):
        tensor[..., position] = _build_dimont_position_tensor(
            position,
            node,
            span,
            full_contexts,
        )

    return tensor, length, span
