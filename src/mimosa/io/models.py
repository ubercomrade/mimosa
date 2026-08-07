"""Model readers: MEME/PFM, BaMM, SiteGA, Dimont, Slim."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from ..errors import InvariantError, ModelFormatError
from ..models import BaMM, Dimont, SiteGA, Slim

MAX_MEME_MOTIF_LENGTH = 10_000
MAX_PFM_LENGTH = 10_000
MAX_LINE_LENGTH = 1_000_000

MAX_BAMM_POSITIONS = 10_000
MAX_BAMM_ORDER = 10
MAX_BAMM_FILE_BYTES = 256 * 1024**2
MAX_BAMM_LINE_LENGTH = 1_000_000
MAX_BAMM_REPRESENTATION_ELEMENTS = 100_000_000
BAMM_EPSILON = 1e-10

MAX_SITEGA_LENGTH = 10_000
SITEGA_EPSILON = 1e-9
DINUC_MAP = {
    "aa": (0, 0), "ac": (0, 1), "ag": (0, 2), "at": (0, 3),
    "ca": (1, 0), "cc": (1, 1), "cg": (1, 2), "ct": (1, 3),
    "ga": (2, 0), "gc": (2, 1), "gg": (2, 2), "gt": (2, 3),
    "ta": (3, 0), "tc": (3, 1), "tg": (3, 2), "tt": (3, 3),
}
DINUC_LIST = [
    "aa", "ac", "ag", "at", "ca", "cc", "cg", "ct",
    "ga", "gc", "gg", "gt", "ta", "tc", "tg", "tt",
]

DIMONT_MAX_LENGTH = 10_000
DIMONT_MAX_SPAN = 10
LOG_UNIFORM_BASE = math.log(4.0)

SLIM_MAX_LENGTH = 10_000
SLIM_MAX_SPAN = 10


def _basename_without_extension(path):
    base = str(path).rsplit("/", 1)[-1]
    dot = base.rfind(".")
    return base if dot < 0 else base[:dot]


def _validate_probability_rows(rows, path, label):
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            if not math.isfinite(value):
                raise ModelFormatError(
                    path, f"{label} contains non-finite value in row {i}, column {j}."
                )
            if not 0.0 <= value <= 1.0:
                raise ModelFormatError(
                    path, f"{label} value in row {i}, column {j} must lie in [0, 1]."
                )
        row_sum = sum(row)
        if not math.isclose(row_sum, 1.0, rel_tol=1e-4, abs_tol=1e-6):
            raise ModelFormatError(
                path, f"{label} row {i} sums to {row_sum}, expected approximately 1.0."
            )


def read_meme(path, index=0):
    if index < 0:
        raise ModelFormatError(path, f"motif index must be non-negative, got {index}.")
    motif_count = 0
    with open(path, "r", encoding="ascii", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("MOTIF"):
                is_target = motif_count == index
                motif_count += 1
                parts = line.strip().split()
                if len(parts) < 2:
                    raise ModelFormatError(path, "MOTIF line has no name.")
                name = parts[1]
                header = f.readline()
                if not header:
                    raise ModelFormatError(path, f"motif {name} has no header.")
                header_parts = header.strip().split()
                try:
                    w_idx = header_parts.index("w=")
                    motif_length = int(header_parts[w_idx + 1])
                except (ValueError, IndexError):
                    raise ModelFormatError(
                        path, f"motif {name} header has no valid 'w=' length field."
                    )
                if is_target:
                    rows = _read_meme_matrix_rows(f, path, name, motif_length)
                    _validate_probability_rows(rows, path, f"motif {name}")
                    pfm = np.array(rows, dtype=np.float32).T
                    return name, pfm
                else:
                    if motif_length <= 0:
                        raise ModelFormatError(path, f"motif {name} has invalid length.")
                    for _ in range(motif_length):
                        if not f.readline():
                            raise ModelFormatError(
                                path, f"motif {name} has fewer rows than declared length."
                            )
    if motif_count == 0:
        raise ModelFormatError(path, "no motifs found.")
    raise ModelFormatError(
        path,
        f"motif index {index} out of range. File contains {motif_count} motifs.",
    )


def _read_meme_matrix_rows(f, path, name, nrows):
    if nrows <= 0:
        raise ModelFormatError(path, f"motif {name} has invalid length.")
    if nrows > MAX_MEME_MOTIF_LENGTH:
        raise ModelFormatError(
            path, f"motif {name} length {nrows} exceeds limit {MAX_MEME_MOTIF_LENGTH}."
        )
    rows = []
    for i in range(nrows):
        line = f.readline()
        if not line:
            raise ModelFormatError(
                path, f"motif {name} has fewer rows than declared length."
            )
        if len(line) > MAX_LINE_LENGTH:
            raise ModelFormatError(path, f"motif {name} row exceeds line length limit.")
        parts = line.strip().split()
        if len(parts) != 4:
            raise ModelFormatError(
                path,
                f"motif {name} row {i} has {len(parts)} columns, expected 4.",
            )
        row = []
        for p in parts:
            try:
                v = float(p)
            except ValueError:
                raise ModelFormatError(
                    path, f"motif {name} row {i} has non-numeric value: {p}."
                )
            row.append(v)
        rows.append(row)
    return rows


def read_pfm(path):
    rows = []
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if len(line) > MAX_LINE_LENGTH:
                raise ModelFormatError(path, "line exceeds length limit.")
            stripped = line.strip()
            if not stripped or stripped.startswith(">"):
                continue
            if len(rows) >= MAX_PFM_LENGTH:
                raise ModelFormatError(path, "PFM row count exceeds the size limit.")
            parts = stripped.split()
            if len(parts) != 4:
                raise ModelFormatError(
                    path, f"PFM row has {len(parts)} columns, expected 4."
                )
            row = []
            for p in parts:
                try:
                    v = float(p)
                except ValueError:
                    raise ModelFormatError(path, f"non-numeric value: {p}.")
                row.append(v)
            rows.append(row)
    if not rows:
        raise ModelFormatError(path, "PFM file is empty.")
    _validate_probability_rows(rows, path, "PFM")
    pfm = np.array(rows, dtype=np.float32).T
    if pfm.shape[1] <= 0:
        raise ModelFormatError(path, "motif length must be positive.")
    if not np.all(np.isfinite(pfm)):
        raise ModelFormatError(path, "matrix contains non-finite values.")
    return _basename_without_extension(path), pfm


# ── BaMM ─────────────────────────────────────────────────────────────────────

def _parse_bamm_blocks(path):
    blocks = []
    current_block = []
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if len(line) > MAX_BAMM_LINE_LENGTH:
                raise ModelFormatError(path, "line exceeds length limit.")
            stripped = line.strip()
            if not stripped:
                if current_block:
                    if len(blocks) >= MAX_BAMM_POSITIONS:
                        raise ModelFormatError(
                            path, f"BaMM has more than {MAX_BAMM_POSITIONS} positions."
                        )
                    blocks.append(current_block)
                    current_block = []
                continue
            if stripped.startswith("#"):
                continue
            parts = stripped.split()
            if not parts:
                continue
            order_index = len(current_block)
            if order_index > MAX_BAMM_ORDER:
                raise ModelFormatError(
                    path,
                    f"BaMM order exceeds the maximum supported order {MAX_BAMM_ORDER}.",
                )
            expected_width = 4 ** (order_index + 1)
            if len(parts) != expected_width:
                raise ModelFormatError(
                    path,
                    f"BaMM order {order_index} width: expected {expected_width}, got {len(parts)}.",
                )
            row = []
            for p in parts:
                try:
                    v = float(p)
                except ValueError:
                    raise ModelFormatError(path, f"non-numeric value: {p}.")
                row.append(v)
            if not all(math.isfinite(v) for v in row):
                raise ModelFormatError(path, "non-finite values in BaMM data.")
            if any(v < 0 for v in row):
                raise ModelFormatError(path, "BaMM values must be non-negative.")
            current_block.append(row)
    if current_block:
        if len(blocks) >= MAX_BAMM_POSITIONS:
            raise ModelFormatError(path, f"BaMM has more than {MAX_BAMM_POSITIONS} positions.")
        blocks.append(current_block)
    if not blocks:
        raise ModelFormatError(path, "no valid position blocks found.")
    n_orders = len(blocks[0])
    for i, block in enumerate(blocks):
        if len(block) != n_orders:
            raise ModelFormatError(
                path,
                f"inconsistent orders in block {i}: expected {n_orders}, got {len(block)}.",
            )
    for pos_idx, block in enumerate(blocks):
        for k, arr in enumerate(block):
            expected_width = 4 ** (k + 1)
            if len(arr) != expected_width:
                raise ModelFormatError(
                    path,
                    f"BaMM order {k - 1} width in block {pos_idx}: expected {expected_width}, got {len(arr)}.",
                )
    return blocks


def _decode_5ary(code, n_digits):
    digits = [0] * n_digits
    remaining = code
    for i in range(n_digits - 1, -1, -1):
        digits[i] = remaining % 5
        remaining //= 5
    return digits


def _build_bamm_representation(blocks, target_order, n_positions, path):
    if not (0 <= target_order <= MAX_BAMM_ORDER):
        raise ModelFormatError(
            path, f"BaMM order must be between 0 and {MAX_BAMM_ORDER}, got {target_order}."
        )
    n_rows = 5 ** (target_order + 1)
    if n_rows * n_positions > MAX_BAMM_REPRESENTATION_ELEMENTS:
        raise ModelFormatError(
            path,
            f"BaMM representation has {n_rows * n_positions} elements, exceeds the limit {MAX_BAMM_REPRESENTATION_ELEMENTS}.",
        )
    rep = np.empty((n_rows, n_positions), dtype=np.float32)
    for pos in range(n_positions):
        current_k = min(pos, target_order)
        p_motif = blocks[pos][current_k]
        expected_width = 4 ** (current_k + 1)
        if len(p_motif) != expected_width:
            raise ModelFormatError(
                path,
                f"position {pos} order {current_k}: expected {expected_width} values, got {len(p_motif)}.",
            )
        uniform_bg = 0.25 ** (current_k + 1)
        log_odds = np.log(
            (np.array(p_motif, dtype=np.float32) + BAMM_EPSILON) / (uniform_bg + BAMM_EPSILON)
        )
        col_min = float(log_odds.min())
        missing_dims = target_order - current_k
        for row in range(n_rows):
            digits = _decode_5ary(row, target_order + 1)
            if 4 in digits:
                rep[row, pos] = col_min
            else:
                relevant = digits[missing_dims:]
                idx = 0
                for d in relevant:
                    idx = idx * 4 + d
                rep[row, pos] = log_odds[idx]
    return rep


def read_bamm(path, order=None):
    blocks = _parse_bamm_blocks(path)
    n_positions = len(blocks)
    max_order = len(blocks[0]) - 1
    requested_order = max_order if order is None else int(order)
    target_order = min(requested_order, max_order)
    if target_order < 0:
        raise ModelFormatError(path, f"order must be non-negative, got {order}.")
    rep = _build_bamm_representation(blocks, target_order, n_positions, path)
    return BaMM(_basename_without_extension(path), rep, target_order, n_positions)


# ── SiteGA ───────────────────────────────────────────────────────────────────

def read_sitega(path):
    with open(path, "r", encoding="ascii", errors="replace") as f:
        lines = f.readlines()
    if not lines:
        raise ModelFormatError(path, "empty file: missing model name.")
    name = lines[0].strip()
    if not name:
        raise ModelFormatError(path, "missing model name.")
    if len(lines) < 2:
        raise ModelFormatError(path, "missing LPD count line.")
    lpd_parts = lines[1].strip().split()
    if not lpd_parts:
        raise ModelFormatError(path, "missing LPD count.")
    try:
        lpd_count = int(lpd_parts[0])
    except ValueError:
        raise ModelFormatError(path, f"invalid LPD count: {lpd_parts[0]}.")
    if lpd_count < 0:
        raise ModelFormatError(path, "LPD count must be non-negative.")
    if len(lines) < 3:
        raise ModelFormatError(path, "missing model length line.")
    len_parts = lines[2].strip().split()
    if not len_parts:
        raise ModelFormatError(path, "missing model length.")
    try:
        motif_length = int(len_parts[0])
    except ValueError:
        raise ModelFormatError(path, f"invalid model length: {len_parts[0]}.")
    if motif_length <= 0:
        raise ModelFormatError(path, f"model length must be positive, got {motif_length}.")
    if motif_length > MAX_SITEGA_LENGTH:
        raise ModelFormatError(
            path, f"model length {motif_length} exceeds limit {MAX_SITEGA_LENGTH}."
        )
    rep = np.zeros((25, motif_length), dtype=np.float32)
    parsed_segments = 0
    for line_idx, line in enumerate(lines[5:]):
        stripped = line.strip()
        if not stripped:
            continue
        parsed_segments += 1
        parts = stripped.split()
        if len(parts) != 5:
            raise ModelFormatError(
                path, f"line {line_idx + 6} must contain 5 fields, got {len(parts)}."
            )
        try:
            start_idx = int(parts[0])
            stop_idx = int(parts[1])
            value = float(parts[2])
        except ValueError:
            raise ModelFormatError(path, f"invalid numeric field on line {line_idx + 6}.")
        dinucleotide = parts[4].lower()
        if dinucleotide not in DINUC_MAP:
            raise ModelFormatError(
                path, f"invalid dinucleotide: {dinucleotide} on line {line_idx + 6}."
            )
        if start_idx < 0 or stop_idx < start_idx or stop_idx >= motif_length:
            raise ModelFormatError(
                path,
                f"range {start_idx}-{stop_idx} is outside model length {motif_length} on line {line_idx + 6}.",
            )
        nuc1, nuc2 = DINUC_MAP[dinucleotide]
        row_code = nuc1 * 5 + nuc2
        n_positions = stop_idx - start_idx + 1
        per_position = np.float32(value / n_positions)
        rep[row_code, start_idx : stop_idx + 1] += per_position
    if parsed_segments != lpd_count:
        raise ModelFormatError(
            path,
            f"LPD count {lpd_count} does not match parsed segment count {parsed_segments}.",
        )
    if not np.all(np.isfinite(rep)):
        raise ModelFormatError(path, "representation contains non-finite values.")
    return SiteGA(name, rep, motif_length)


def write_sitega(path, model):
    if not isinstance(model, SiteGA):
        raise InvariantError(
            f"write_sitega requires a SiteGA model, got {type(model).__name__}."
        )
    rep = model.weights
    motif_length = model.motif_length
    mn = float(rep.min(axis=0).sum())
    mx = float(rep.max(axis=0).sum())
    segments = []
    for nuc1 in range(4):
        for nuc2 in range(4):
            row_code = nuc1 * 5 + nuc2
            row_data = rep[row_code]
            if np.all(np.abs(row_data) <= SITEGA_EPSILON):
                continue
            dinucleotide = DINUC_LIST[nuc1 * 4 + nuc2]
            pos = 0
            while pos < motif_length:
                while pos < motif_length and abs(row_data[pos]) <= SITEGA_EPSILON:
                    pos += 1
                if pos >= motif_length:
                    break
                start_pos = pos
                current_val = row_data[pos]
                while pos + 1 < motif_length and abs(row_data[pos + 1] - current_val) < SITEGA_EPSILON:
                    pos += 1
                segments.append((start_pos, pos, float(current_val), dinucleotide))
                pos += 1
    dinuc_index = {d: i for i, d in enumerate(DINUC_LIST)}
    with open(path, "w", encoding="ascii") as f:
        f.write(f"{model.name}\n")
        f.write(f"{len(segments)}\tLPD count\n")
        f.write(f"{motif_length}\tModel length\n")
        f.write(f"{mn:.12f}\tMinimum\n")
        f.write(f"{mx:.12f}\tRazmah\n")
        for start, stop, val, dinuc in segments:
            range_length = stop - start + 1
            total_value = val * range_length
            f.write(f"{start}\t{stop}\t{total_value:.12f}\t{dinuc_index[dinuc]}\t{dinuc}\n")


# ── Dimont ───────────────────────────────────────────────────────────────────

class _DimontTreeNode:
    __slots__ = ("context_pos", "scores", "children")

    def __init__(self, context_pos, scores, children):
        self.context_pos = context_pos
        self.scores = scores
        self.children = children


def _xml_numeric(elem, path, label):
    if elem is None:
        raise ModelFormatError(path, f"malformed XML: missing {label}.")
    text = " ".join(
        t for t in (elem.text, *(c.tail or "" for c in elem)) if t and t.strip()
    )
    tokens = text.split()
    for tok in reversed(tokens):
        try:
            return float(tok)
        except ValueError:
            continue
    raise ModelFormatError(path, f"malformed XML: no numeric value for {label}.")


def _xml_find(elem, path):
    return elem.find("." + path if path.startswith("/") else path)


def _parse_tree_element(elem, path):
    ctx_pos = int(_xml_numeric(_xml_find(elem, "contextPos"), path, "Dimont tree contextPos"))
    pars = _xml_find(elem, "pars")
    if pars is not None:
        pars_pos_children = [c for c in pars if c.tag == "pos"]
        if pars_pos_children:
            scores = [-math.inf] * 4
            for par_pos in pars_pos_children:
                par = _xml_find(par_pos, "parameter")
                if par is None:
                    raise ModelFormatError(path, "malformed Dimont tree: missing parameter.")
                symbol = int(_xml_numeric(_xml_find(par, "symbol"), path, "Dimont tree parameter symbol"))
                value = _xml_numeric(_xml_find(par, "value"), path, "Dimont tree parameter value")
                if symbol < 0 or symbol > 3:
                    raise ModelFormatError(path, f"Dimont tree parameter symbol out of range: {symbol}.")
                scores[symbol] = value
            return _DimontTreeNode(ctx_pos, scores, None)
    children_elem = _xml_find(elem, "children")
    if children_elem is not None:
        children = [None] * 4
        for child_pos in children_elem:
            if child_pos.tag != "pos":
                continue
            child_idx = int(child_pos.get("val"))
            if child_idx < 0 or child_idx > 3:
                raise ModelFormatError(path, f"Dimont tree child index out of range: {child_idx}.")
            child_elem = _xml_find(child_pos, "treeElement")
            if child_elem is None:
                raise ModelFormatError(path, "malformed Dimont tree: missing child treeElement.")
            children[child_idx] = _parse_tree_element(child_elem, path)
        if any(c is None for c in children):
            raise ModelFormatError(path, "malformed Dimont tree: expected 4 children.")
        return _DimontTreeNode(ctx_pos, None, children)
    raise ModelFormatError(path, "malformed Dimont tree: expected pars or children.")


def _dimont_span(context_positions_list):
    span = 0
    for position, positions in enumerate(context_positions_list):
        if positions:
            span = max(span, max(position - p for p in positions))
    return span


def _iter_contexts(span):
    if span == 0:
        return [()]
    import itertools

    return list(itertools.product(range(4), repeat=span))


def _context_value(full_context, span, position, absolute_position, path):
    if absolute_position < 0:
        raise ModelFormatError(
            path,
            f"model references position {absolute_position} before the motif start at position {position}.",
        )
    axis = absolute_position - (position - span)
    if axis < 0 or axis >= span:
        raise ModelFormatError(
            path,
            f"context position {absolute_position} at motif position {position} does not fit span {span}.",
        )
    return full_context[axis]


def _fill_n_states(rep, pos_idx, span):
    n_rows = rep.shape[0]
    for axis in range(span):
        for row in range(n_rows):
            digits = _decode_5ary(row, span + 1)
            if digits[axis] != 4 or digits[span] == 4:
                continue
            min_val = math.inf
            for nt in range(4):
                probe = digits.copy()
                probe[axis] = nt
                probe_code = 0
                for d in probe:
                    probe_code = probe_code * 5 + d
                val = rep[probe_code, pos_idx]
                if val < min_val:
                    min_val = val
            rep[row, pos_idx] = min_val
    for row in range(n_rows):
        digits = _decode_5ary(row, span + 1)
        if digits[span] != 4:
            continue
        min_val = math.inf
        for nt in range(4):
            probe = digits.copy()
            probe[span] = nt
            probe_code = 0
            for d in probe:
                probe_code = probe_code * 5 + d
            val = rep[probe_code, pos_idx]
            if val < min_val:
                min_val = val
        rep[row, pos_idx] = min_val


def _build_position_column(context_scores, span, pos_idx, rep):
    n_rows = 5 ** (span + 1)
    for row in range(n_rows):
        digits = _decode_5ary(row, span + 1)
        if 4 in digits:
            rep[row, pos_idx] = 0.0
        else:
            ctx = tuple(digits[:span])
            symbol = digits[span]
            rep[row, pos_idx] = np.float32(context_scores[ctx][symbol])
    _fill_n_states(rep, pos_idx, span)


def read_dimont(path):
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ModelFormatError(path, f"malformed XML: {exc}.") from exc
    root = tree.getroot()
    model = _xml_find(root, "//MarkovModelDiffSM")
    if model is None:
        raise ModelFormatError(path, "could not find Dimont motif model (MarkovModelDiffSM).")
    trees = _xml_find(model, "bayesianNetworkSF/trees")
    if trees is None:
        raise ModelFormatError(path, "malformed Dimont model: missing bayesianNetworkSF/trees.")
    context_positions_list = []
    nodes = []
    for pos_elem in trees:
        if pos_elem.tag != "pos":
            continue
        pt = _xml_find(pos_elem, "parameterTree")
        if pt is None:
            raise ModelFormatError(path, "malformed Dimont model: missing parameterTree.")
        ctx_positions = []
        ctx_poss = _xml_find(pt, "contextPoss")
        if ctx_poss is not None:
            for cp_child in ctx_poss:
                if cp_child.tag == "pos":
                    ctx_positions.append(int(_xml_numeric(cp_child, path, "Dimont contextPoss position")))
        root_elem = _xml_find(pt, "root/treeElement")
        if root_elem is None:
            raise ModelFormatError(path, "malformed Dimont model: missing root treeElement.")
        nodes.append(_parse_tree_element(root_elem, path))
        context_positions_list.append(ctx_positions)
    if not nodes:
        raise ModelFormatError(path, "no parameter trees found in Dimont model.")
    length_val = len(nodes)
    if length_val > DIMONT_MAX_LENGTH:
        raise ModelFormatError(path, f"Dimont length {length_val} exceeds limit {DIMONT_MAX_LENGTH}.")
    span = _dimont_span(context_positions_list)
    if span > DIMONT_MAX_SPAN:
        raise ModelFormatError(path, f"Dimont span {span} exceeds limit {DIMONT_MAX_SPAN}.")
    n_rows = 5 ** (span + 1)
    rep = np.empty((n_rows, length_val), dtype=np.float32)
    full_contexts = _iter_contexts(span)
    for pos_idx, node in enumerate(nodes):
        pt_pos = pos_idx
        context_scores = {}
        for ctx in full_contexts:
            current = node
            while current.scores is None:
                parent_pos = current.context_pos
                ctx_val = _context_value(ctx, span, pt_pos, parent_pos, path)
                current = current.children[ctx_val]
            context_scores[ctx] = [s + LOG_UNIFORM_BASE for s in current.scores]
        _build_position_column(context_scores, span, pos_idx, rep)
    return Dimont(_basename_without_extension(path), rep, span, length_val)


# ── Slim ─────────────────────────────────────────────────────────────────────

def _slim_pos_children(elem):
    return [c for c in elem if c.tag == "pos"]


def _slim_parse_component_params(slim, path):
    elem = _xml_find(slim, "componentMixtureParameters")
    if elem is None:
        raise ModelFormatError(path, "malformed Slim model: missing componentMixtureParameters.")
    result = []
    for pos_p in _slim_pos_children(elem):
        comps = _slim_pos_children(pos_p)
        if not comps:
            raise ModelFormatError(path, "malformed Slim model: empty component mixture.")
        result.append([_xml_numeric(c, path, "Slim component weight") for c in comps])
    if not result:
        raise ModelFormatError(path, "malformed Slim model: no component positions.")
    return result


def _slim_parse_ancestor_params(slim, path):
    elem = _xml_find(slim, "ancestorMixtureParameters")
    if elem is None:
        raise ModelFormatError(path, "malformed Slim model: missing ancestorMixtureParameters.")
    result = []
    for pos_p in _slim_pos_children(elem):
        comp_list = []
        for comp_c in _slim_pos_children(pos_p):
            ancs = _slim_pos_children(comp_c)
            if not ancs:
                raise ModelFormatError(path, "malformed Slim model: empty ancestor mixture.")
            comp_list.append([_xml_numeric(a, path, "Slim ancestor weight") for a in ancs])
        result.append(comp_list)
    return result


def _slim_parse_dependency_params(slim, path):
    elem = _xml_find(slim, "dependencyParameters")
    if elem is None:
        raise ModelFormatError(path, "malformed Slim model: missing dependencyParameters.")
    result = []
    for pos_p in _slim_pos_children(elem):
        comp_list = []
        for comp_c in _slim_pos_children(pos_p):
            rows = _slim_pos_children(comp_c)
            if not rows:
                raise ModelFormatError(path, "malformed Slim model: empty dependency rows.")
            row_list = []
            for row_r in rows:
                syms = _slim_pos_children(row_r)
                if not syms:
                    raise ModelFormatError(path, "malformed Slim model: empty dependency row.")
                row_list.append([_xml_numeric(s, path, "Slim dependency value") for s in syms])
            comp_list.append(row_list)
        result.append(comp_list)
    return result


def _slim_span(component_raw, ancestor_raw, path):
    span = 0
    for position in range(len(component_raw)):
        n_components = len(component_raw[position])
        for component_index in range(1, n_components):
            if component_index >= len(ancestor_raw[position]):
                raise ModelFormatError(
                    path,
                    f"malformed Slim model: missing ancestor mixture for position {position} component {component_index}.",
                )
            ancestor_count = len(ancestor_raw[position][component_index])
            if ancestor_count <= 0:
                raise ModelFormatError(
                    path,
                    f"malformed Slim model: empty ancestor mixture at position {position}.",
                )
            reach = component_index + ancestor_count - 1
            span = max(span, reach)
    return span


def _logsumexp(v):
    m = max(v)
    if not math.isfinite(m):
        return m
    s = sum(math.exp(x - m) for x in v)
    return m + math.log(s)


def _log_normalize(v):
    lse = _logsumexp(v)
    return [x - lse for x in v]


def _slim_symbol_log_probs(position, symbol, full_context, params, path):
    comp_lp = params[0][position]
    dep_lp = params[1][position]
    anc_lp = params[2][position]
    span = params[3]
    alphabet = params[5]
    n_comp = len(comp_lp)
    local_scores = [0.0] * n_comp
    local_scores[0] = comp_lp[0] + dep_lp[0][0][symbol]
    for component_index in range(1, n_comp):
        ancestor_count = len(anc_lp[component_index])
        context_index = 0
        for current_order in range(1, component_index):
            parent_position = position - current_order
            ctx_val = _context_value(full_context, span, position, parent_position, path)
            context_index = context_index * alphabet + ctx_val
        dependency = dep_lp[component_index]
        total_contexts = len(dependency)
        width = len(dependency[0])
        ancestor_scores = [0.0] * ancestor_count
        for ancestor_index in range(ancestor_count):
            parent_position = position - component_index - ancestor_index
            ctx_val = _context_value(full_context, span, position, parent_position, path)
            context_index = (context_index * width + ctx_val + 1) % total_contexts - 1
            ancestor_scores[ancestor_index] = (
                anc_lp[component_index][ancestor_index] + dependency[context_index][symbol]
            )
        local_scores[component_index] = comp_lp[component_index] + _logsumexp(ancestor_scores)
    return _logsumexp(local_scores) + LOG_UNIFORM_BASE


def read_slim(path):
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ModelFormatError(path, f"malformed XML: {exc}.") from exc
    root = tree.getroot()
    slim = _xml_find(root, "//SLIM")
    if slim is None:
        raise ModelFormatError(path, "could not find Slim motif model (SLIM).")
    length_val = int(_xml_numeric(_xml_find(slim, "length"), path, "Slim length"))
    if length_val > SLIM_MAX_LENGTH:
        raise ModelFormatError(path, f"Slim length {length_val} exceeds limit {SLIM_MAX_LENGTH}.")
    component_raw = _slim_parse_component_params(slim, path)
    ancestor_raw = _slim_parse_ancestor_params(slim, path)
    dependency_raw = _slim_parse_dependency_params(slim, path)
    if len(component_raw) != length_val:
        raise ModelFormatError(
            path,
            f"Slim componentMixtureParameters length ({len(component_raw)}) does not match length ({length_val}).",
        )
    span = _slim_span(component_raw, ancestor_raw, path)
    if span > SLIM_MAX_SPAN:
        raise ModelFormatError(path, f"Slim span {span} exceeds limit {SLIM_MAX_SPAN}.")
    if not dependency_raw or not dependency_raw[0] or not dependency_raw[0][0] or not dependency_raw[0][0][0]:
        raise ModelFormatError(path, "malformed Slim model: empty dependency parameters.")
    alphabet = len(dependency_raw[0][0][0])
    comp_lp = [_log_normalize(component_raw[p]) for p in range(length_val)]
    anc_lp = [
        [_log_normalize(ancestor_raw[p][c]) for c in range(len(ancestor_raw[p]))]
        for p in range(length_val)
    ]
    dep_lp = []
    for p in range(length_val):
        comp_dep = []
        for c in range(len(dependency_raw[p])):
            rows = dependency_raw[p][c]
            n_cols = len(rows[0])
            mat = []
            for r in rows:
                if len(r) != n_cols:
                    raise ModelFormatError("", "inconsistent dependency row lengths.")
                mat.append(_log_normalize(r))
            comp_dep.append(mat)
        dep_lp.append(comp_dep)
    params = (comp_lp, dep_lp, anc_lp, span, length_val, alphabet)
    n_rows = 5 ** (span + 1)
    rep = np.empty((n_rows, length_val), dtype=np.float32)
    full_contexts = _iter_contexts(span)
    for position in range(length_val):
        context_scores = {}
        for ctx in full_contexts:
            context_scores[ctx] = [
                _slim_symbol_log_probs(position, symbol, ctx, params, path) for symbol in range(4)
            ]
        _build_position_column(context_scores, span, position, rep)
    return Slim(_basename_without_extension(path), rep, span, length_val)
