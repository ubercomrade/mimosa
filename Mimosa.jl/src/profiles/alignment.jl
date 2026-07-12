# Profile window alignment: shift-based comparison with anchor realignment.

"""
    _window_fits(pos::Int, len::Int, radius::Int)

Check if a window of radius `radius` centered at 1-based position `pos` fits
within `[1, len]`. Equivalent to Python's `_window_fits` (0-based: `pos - radius >= 0 && pos + radius < length`).
"""
function _window_fits(pos::Int, len::Int, radius::Int)
    return pos - radius >= 1 && pos + radius <= len
end

"""
    _realign_query_position(r::AbstractVector{Float32}, expected::Int, radius::Int)

Find the best-scoring position in `[max(1, expected-radius), min(len, expected+radius)]`.
Returns 0 if the range is empty. Matches Python's `_realign_query_position`.
"""
function _realign_query_position(r::AbstractVector{Float32}, expected::Int, radius::Int)
    len = length(r)
    left = max(1, expected - radius)
    right = min(len, expected + radius)
    left > right && return 0
    best_pos = left
    best_score = r[left]
    @inbounds for pos in (left + 1):right
        if r[pos] > best_score
            best_score = r[pos]
            best_pos = pos
        end
    end
    return best_pos
end

# Collect unique candidate positions for one row from both query and target
# anchors. Returns a vector of unique 1-based positions in query coordinates.
function _collect_row_candidates(
    r1::AbstractVector{Float32},
    len1::Int,
    len2::Int,
    query_csr::AnchorCSR,
    target_csr::AnchorCSR,
    row::Int,
    shift::Int,
    window_radius::Int,
    realign_window::Int,
)
    candidates = Int[]
    seen = falses(len1)

    # Query anchors
    @inbounds for idx in query_csr.offsets[row]:(query_csr.offsets[row + 1] - 1)
        pos1 = query_csr.positions[idx]
        pos2 = pos1 + shift
        if _window_fits(pos1, len1, window_radius) &&
            _window_fits(pos2, len2, window_radius)
            if !seen[pos1]
                seen[pos1] = true
                push!(candidates, pos1)
            end
        end
    end

    # Target anchors (realigned to query coordinates)
    @inbounds for idx in target_csr.offsets[row]:(target_csr.offsets[row + 1] - 1)
        expected_pos1 = target_csr.positions[idx] - shift
        pos1 = _realign_query_position(r1, expected_pos1, realign_window)
        pos1 == 0 && continue
        pos2 = pos1 + shift
        if _window_fits(pos1, len1, window_radius) &&
            _window_fits(pos2, len2, window_radius)
            if !seen[pos1]
                seen[pos1] = true
                push!(candidates, pos1)
            end
        end
    end

    return candidates
end

# Accumulate pooled overlap sums (for CO and Dice metrics).
function _accumulate_pooled(
    r1::AbstractVector{Float32},
    r2::AbstractVector{Float32},
    candidates::Vector{Int},
    shift::Int,
    window_radius::Int,
)
    sum1 = 0.0
    sum2 = 0.0
    intersection = 0.0
    @inbounds for pos1 in candidates
        pos2 = pos1 + shift
        for offset in (-window_radius):window_radius
            v1 = Float64(r1[pos1 + offset])
            v2 = Float64(r2[pos2 + offset])
            sum1 += v1
            sum2 += v2
            intersection += min(v1, v2)
        end
    end
    return (sum1, sum2, intersection)
end

# Accumulate rowwise overlap-based values (for CO_rowwise and Dice_rowwise).
function _accumulate_rowwise_overlap(
    r1::AbstractVector{Float32},
    r2::AbstractVector{Float32},
    candidates::Vector{Int},
    shift::Int,
    window_radius::Int,
    use_dice::Bool,
)
    score_sum = 0.0
    finite_count = 0
    @inbounds for pos1 in candidates
        pos2 = pos1 + shift
        sum1 = 0.0
        sum2 = 0.0
        intersection = 0.0
        for offset in (-window_radius):window_radius
            v1 = Float64(r1[pos1 + offset])
            v2 = Float64(r2[pos2 + offset])
            sum1 += v1
            sum2 += v2
            intersection += min(v1, v2)
        end
        denom = use_dice ? sum1 + sum2 : min(sum1, sum2)
        if denom > PROFILE_EPS
            val = use_dice ? 2.0 * intersection / denom : intersection / denom
            score_sum += val
            finite_count += 1
        end
    end
    return (score_sum, finite_count)
end

# Accumulate rowwise cosine values.
function _accumulate_cosine(
    r1::AbstractVector{Float32},
    r2::AbstractVector{Float32},
    candidates::Vector{Int},
    shift::Int,
    window_radius::Int,
)
    score_sum = 0.0
    finite_count = 0
    @inbounds for pos1 in candidates
        pos2 = pos1 + shift
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for offset in (-window_radius):window_radius
            v1 = Float64(r1[pos1 + offset])
            v2 = Float64(r2[pos2 + offset])
            dot += v1 * v2
            norm1 += v1 * v1
            norm2 += v2 * v2
        end
        denom = sqrt(norm1) * sqrt(norm2)
        if denom > PROFILE_EPS
            score_sum += dot / denom
            finite_count += 1
        end
    end
    return (score_sum, finite_count)
end

"""
    score_shift(scores1, scores2, query_csr, target_csr, shift, window_radius, realign_window, metric)

Evaluate one shift across all rows and reduce row-level partials into a
`(score::Float32, n_sites::Int)` tuple. Matches Python's `score_shift` serial path.

- `scores1`: query strand scores (RaggedArray, 1-based positions per row).
- `scores2`: target strand scores.
- `query_csr`/`target_csr`: anchor CSR structures for this orientation pair.
- `shift`: target position = query position + shift.
- `window_radius`: half-window size (0 = single position).
- `realign_window`: realignment search radius for target anchors.
- `metric`: typed profile metric.
"""
function score_shift(
    scores1::RaggedArray{Float32},
    scores2::RaggedArray{Float32},
    query_csr::AnchorCSR,
    target_csr::AnchorCSR,
    shift::Int,
    window_radius::Int,
    realign_window::Int,
    metric::AbstractProfileMetric,
)
    n = nrows(scores1)
    total_sum1 = 0.0
    total_sum2 = 0.0
    total_intersection = 0.0
    total_row_score = 0.0
    total_finite = 0
    total_sites = 0

    for r in 1:n
        len1 = rowlength(scores1, r)
        len2 = rowlength(scores2, r)

        r1 = row(scores1, r)
        r2 = row(scores2, r)

        candidates = _collect_row_candidates(
            r1, len1, len2, query_csr, target_csr, r, shift, window_radius, realign_window
        )
        count = length(candidates)
        total_sites += count

        if count == 0
            continue
        end

        if is_pooled(metric)
            s1, s2, inter = _accumulate_pooled(r1, r2, candidates, shift, window_radius)
            total_sum1 += s1
            total_sum2 += s2
            total_intersection += inter
        elseif metric isa CosineSimilarityProfile
            s_sum, f_count = _accumulate_cosine(r1, r2, candidates, shift, window_radius)
            total_row_score += s_sum
            total_finite += f_count
        else
            use_dice = is_dice_metric(metric)
            s_sum, f_count = _accumulate_rowwise_overlap(
                r1, r2, candidates, shift, window_radius, use_dice
            )
            total_row_score += s_sum
            total_finite += f_count
        end
    end

    if total_sites == 0
        return (0.0f0, 0)
    end

    if is_pooled(metric)
        if is_dice_metric(metric)
            denom = total_sum1 + total_sum2
            score = denom > PROFILE_EPS ? 2.0 * total_intersection / denom : 0.0
        else
            denom = min(total_sum1, total_sum2)
            score = denom > PROFILE_EPS ? total_intersection / denom : 0.0
        end
    else
        score = total_finite == 0 ? 0.0 : total_row_score / total_finite
    end

    return (Float32(score), total_sites)
end

# ── Orientation scoring ──────────────────────────────────────────────────

# Profile orientation pairs: (label, query_strand, target_strand).
# Strand indices: 1=forward, 2=reverse (1-based Julia convention).
const PROFILE_ORIENTATION_PAIRS = (("++", 1, 1), ("--", 2, 2), ("+-", 1, 2), ("-+", 2, 1))

"""
    _score_orientation_pair(query_bundle, target_bundle, query_anchors, target_anchors, search_range, window_radius, realign_window, metric)

Score one orientation pair across all shifts in `[-search_range, search_range]`.
Returns `(score, shift, n_sites, orientation_label)`.

Tie-breaking (matching Python):
1. Higher score wins.
2. On equal score: more n_sites wins.
3. On equal n_sites: smaller |shift| wins.
4. First in iteration order wins on complete tie.
"""
function _score_orientation_pair(
    query_bundle::StrandPair{<:RaggedArray{Float32}},
    target_bundle::StrandPair{<:RaggedArray{Float32}},
    query_anchors::AnchorCSR,
    target_anchors::AnchorCSR,
    query_strand::Int,
    target_strand::Int,
    orientation_label::String,
    search_range::Int,
    window_radius::Int,
    realign_window::Int,
    metric::AbstractProfileMetric,
)
    query_scores = query_strand == 1 ? query_bundle.forward : query_bundle.reverse
    target_scores = target_strand == 1 ? target_bundle.forward : target_bundle.reverse

    best_score = 0.0f0
    best_shift = 0
    best_n_sites = 0

    for shift in (-search_range):search_range
        score, n_sites = score_shift(
            query_scores,
            target_scores,
            query_anchors,
            target_anchors,
            shift,
            window_radius,
            realign_window,
            metric,
        )
        if Float64(score) > Float64(best_score) || (
            Float64(score) == Float64(best_score) && (
                n_sites > best_n_sites ||
                (n_sites == best_n_sites && abs(shift) < abs(best_shift))
            )
        )
            best_score = score
            best_shift = shift
            best_n_sites = n_sites
        end
    end

    return (best_score, best_shift, best_n_sites, orientation_label)
end

"""
    ProfileConfig

Configuration for profile comparison.

Fields:
- `metric::AbstractProfileMetric`: profile metric type.
- `search_range::Int`: maximum shift to search (default 10).
- `window_radius::Int`: half-window size for site windows (default 10).
- `realign_window::Int`: realignment search radius (default 3).
- `min_logfpr::Float32`: minimum log FPR for threshold anchors (0 = best anchors).
"""
Base.@kwdef struct ProfileConfig{M<:AbstractProfileMetric}
    metric::M = OverlapCoefficient()
    search_range::Int = 10
    window_radius::Int = 10
    realign_window::Int = 3
    min_logfpr::Float32 = Float32(0.0)
end

"""
    profile_compare(query_bundle, query_anchors, target_bundle, target_anchors, config::ProfileConfig)

Compare two normalized profile bundles using pre-collected anchor CSRs and
return
`(score::Float32, offset::Int, orientation::String, n_sites::Int, metric_name::String)`.

`query_anchors` and `target_anchors` are `(forward_csr, reverse_csr)` tuples.

Scores all four orientation pairs (`++`, `--`, `+-`, `-+`) and selects the best
with deterministic tie-breaking per ADR 0006: higher score wins, then
orientation priority `++ > +- > -+ > --`.
"""
function profile_compare(
    query_bundle::StrandPair{<:RaggedArray{Float32}},
    query_anchors::Tuple{AnchorCSR,AnchorCSR},
    target_bundle::StrandPair{<:RaggedArray{Float32}},
    target_anchors::Tuple{AnchorCSR,AnchorCSR},
    config::ProfileConfig,
)
    metric = config.metric

    # Score all four orientation pairs
    best_score = 0.0f0
    best_shift = 0
    best_n_sites = 0
    best_orientation = "++"
    best_rank = 0

    for (i, (label, q_strand, t_strand)) in enumerate(PROFILE_ORIENTATION_PAIRS)
        qa = q_strand == 1 ? query_anchors[1] : query_anchors[2]
        ta = t_strand == 1 ? target_anchors[1] : target_anchors[2]
        result = _score_orientation_pair(
            query_bundle,
            target_bundle,
            qa,
            ta,
            q_strand,
            t_strand,
            label,
            config.search_range,
            config.window_radius,
            config.realign_window,
            metric,
        )
        score, shift, n_sites = result

        # Tie-breaking: higher score, then orientation priority (++,+-,-+,--)
        rank = i - 1  # 0-indexed rank
        if Float64(score) > Float64(best_score) ||
            (Float64(score) == Float64(best_score) && rank < best_rank)
            best_score = score
            best_shift = shift
            best_n_sites = n_sites
            best_orientation = label
            best_rank = rank
        end
    end

    return (best_score, best_shift, best_orientation, best_n_sites, metric_name(metric))
end

"""
    profile_compare(query_bundle, target_bundle, config::ProfileConfig)

Compare two normalized profile bundles and return
`(score::Float32, offset::Int, orientation::String, n_sites::Int, metric_name::String)`.

Collects anchors for both bundles internally. For one-to-many comparison where
the query anchors should be reused, use the variant with pre-computed anchors.

Scores all four orientation pairs (`++`, `--`, `+-`, `-+`) and selects the best
with deterministic tie-breaking per ADR 0006: higher score wins, then
orientation priority `++ > +- > -+ > --`.
"""
function profile_compare(
    query_bundle::StrandPair{<:RaggedArray{Float32}},
    target_bundle::StrandPair{<:RaggedArray{Float32}},
    config::ProfileConfig,
)
    threshold = config.min_logfpr
    query_anchors = _collect_both_anchors(query_bundle, threshold)
    target_anchors = _collect_both_anchors(target_bundle, threshold)
    return profile_compare(
        query_bundle, query_anchors, target_bundle, target_anchors, config
    )
end

# ── Prepared profile (one-to-many reuse) ───────────────────────────────────────

"""
    PreparedProfile

A pre-normalized profile bundle with pre-collected anchors, ready for
repeated comparison against multiple targets without re-computing the
query side.

Fields:
- `name::String`: profile name.
- `bundle::StrandPair{T}`: normalized scores.
- `anchors::Tuple{AnchorCSR,AnchorCSR}`: `(forward, reverse)` anchor CSRs.
"""
struct PreparedProfile{T}
    name::String
    bundle::T
    anchors::Tuple{AnchorCSR,AnchorCSR}
end

"""
    _collect_both_anchors(bundle::StrandPair{<:RaggedArray{Float32}}, threshold::Float32)

Collect anchors for both strands and return `(forward_csr, reverse_csr)`.
"""
function _collect_both_anchors(
    bundle::StrandPair{<:RaggedArray{Float32}}, threshold::Float32
)
    n_rows = nrows(bundle.forward)
    fwd_rows, fwd_pos = collect_anchors(bundle.forward, threshold)
    fwd_csr = build_anchor_csr(fwd_rows, fwd_pos, n_rows)
    rev_rows, rev_pos = collect_anchors(bundle.reverse, threshold)
    rev_csr = build_anchor_csr(rev_rows, rev_pos, n_rows)
    return (fwd_csr, rev_csr)
end

"""
    prepare_profile(model::ScoreProfile; min_logfpr::Float32=0.0f0)

Prepare a [`ScoreProfile`](@ref) for repeated comparison: fit normalization
from the profile's own scores, apply it, and collect anchors for both
strands. Returns a [`PreparedProfile`](@ref).

Keyword arguments:
- `min_logfpr::Float32=0.0`: minimum log FPR for threshold anchors (0 = best anchors).
"""
function prepare_profile(model::ScoreProfile; min_logfpr::Float32=Float32(0.0))
    raw = profile_bundle(model)
    flat = flatten_bundle(raw)
    table = fit(EmpiricalLogTail(), flat)
    norm_bundle = normalize_bundle(table, raw)
    anchors = _collect_both_anchors(norm_bundle, min_logfpr)
    return PreparedProfile(model.name, norm_bundle, anchors)
end

"""
    prepare_profile(model::AbstractMotifModel, sequences::EncodedSequenceBatch;
                    background=nothing, min_logfpr=0.0f0)

Scan a motif model against `sequences` to produce a normalized profile bundle
with pre-collected anchors, ready for repeated comparison. The normalization
is fitted from `background` (falls back to `sequences` when `background=nothing`).

Returns a [`PreparedProfile`](@ref).
"""
function prepare_profile(
    model::AbstractMotifModel,
    sequences::EncodedSequenceBatch;
    background::Union{EncodedSequenceBatch,Nothing}=nothing,
    min_logfpr::Float32=Float32(0.0),
)
    raw = scan(model, sequences; strands=BothStrands())
    bg = background === nothing ? sequences : background
    bg_raw = scan(model, bg; strands=BothStrands())
    flat = flatten_bundle(bg_raw)
    table = fit(EmpiricalLogTail(), flat)
    norm_bundle = normalize_bundle(table, raw)
    anchors = _collect_both_anchors(norm_bundle, min_logfpr)
    return PreparedProfile(model.name, norm_bundle, anchors)
end

# ── PreparedProfile compare methods ────────────────────────────────────────────

"""
    compare(query::PreparedProfile, target::ScoreProfile; metric=:co, kwargs...)

Compare a [`PreparedProfile`](@ref) (pre-normalized, pre-anchored) against a
[`ScoreProfile`](@ref) target. The query normalization and anchor collection
are reused; only the target is prepared.

Keyword arguments:
- `metric`: profile metric (`:co`, `:co_rowwise`, `:dice`, `:dice_rowwise`,
  `:cosine`, or a typed `AbstractProfileMetric`). Default `:co`.
- `search_range::Int=10`, `window_radius::Int=10`, `realign_window::Int=3`,
  `min_logfpr::Float32=0.0`.
"""
function compare(
    query::PreparedProfile,
    target::ScoreProfile;
    metric::Union{AbstractString,Symbol,AbstractProfileMetric}=:co,
    search_range::Int=10,
    window_radius::Int=10,
    realign_window::Int=3,
    min_logfpr::Float32=Float32(0.0),
)
    m = _resolve_profile_metric(metric)
    target_raw = profile_bundle(target)
    target_flat = flatten_bundle(target_raw)
    target_table = fit(EmpiricalLogTail(), target_flat)
    target_norm = normalize_bundle(target_table, target_raw)
    target_anchors = _collect_both_anchors(target_norm, min_logfpr)
    config = ProfileConfig(;
        metric=m,
        search_range=search_range,
        window_radius=window_radius,
        realign_window=realign_window,
        min_logfpr=min_logfpr,
    )
    score, shift, orientation, n_sites, metric_str = profile_compare(
        query.bundle, query.anchors, target_norm, target_anchors, config
    )
    return ComparisonResult(
        query.name, target.name, score, shift, orientation, metric_str, n_sites
    )
end

"""
    compare(query::PreparedProfile, targets::Vector{ScoreProfile}; kwargs...)

One-to-many comparison: compare a single prepared query against multiple
targets, reusing the query's normalized bundle and pre-collected anchors.
Returns a `Vector{ComparisonResult}`.

Each target is prepared (normalized, anchors collected) independently.
"""
function compare(query::PreparedProfile, targets::Vector{ScoreProfile}; kwargs...)
    return [compare(query, t; kwargs...) for t in targets]
end

"""
    compare(query::PreparedProfile, target::AbstractMotifModel,
            sequences::EncodedSequenceBatch; metric=:co, kwargs...)

Compare a [`PreparedProfile`](@ref) against a motif model target by scanning
the target against `sequences` and comparing profiles. The query's normalized
bundle and anchors are reused.
"""
function compare(
    query::PreparedProfile,
    target::AbstractMotifModel,
    sequences::EncodedSequenceBatch;
    metric::Union{AbstractString,Symbol,AbstractProfileMetric}=:co,
    search_range::Int=10,
    window_radius::Int=10,
    realign_window::Int=3,
    min_logfpr::Float32=Float32(0.0),
    background::Union{EncodedSequenceBatch,Nothing}=nothing,
)
    m = _resolve_profile_metric(metric)
    target_norm = _resolve_profile_bundle(target, sequences, background)
    target_anchors = _collect_both_anchors(target_norm, min_logfpr)
    config = ProfileConfig(;
        metric=m,
        search_range=search_range,
        window_radius=window_radius,
        realign_window=realign_window,
        min_logfpr=min_logfpr,
    )
    score, shift, orientation, n_sites, metric_str = profile_compare(
        query.bundle, query.anchors, target_norm, target_anchors, config
    )
    return ComparisonResult(
        query.name, target.name, score, shift, orientation, metric_str, n_sites
    )
end

"""
    compare(query::AbstractMotifModel, target::PreparedProfile,
            sequences::EncodedSequenceBatch; metric=:co, kwargs...)

Compare a motif model query (scanned against `sequences`) against a
[`PreparedProfile`](@ref) target. The target's normalized bundle and anchors
are reused.
"""
function compare(
    query::AbstractMotifModel,
    target::PreparedProfile,
    sequences::EncodedSequenceBatch;
    metric::Union{AbstractString,Symbol,AbstractProfileMetric}=:co,
    search_range::Int=10,
    window_radius::Int=10,
    realign_window::Int=3,
    min_logfpr::Float32=Float32(0.0),
    background::Union{EncodedSequenceBatch,Nothing}=nothing,
)
    m = _resolve_profile_metric(metric)
    query_norm = _resolve_profile_bundle(query, sequences, background)
    query_anchors = _collect_both_anchors(query_norm, min_logfpr)
    config = ProfileConfig(;
        metric=m,
        search_range=search_range,
        window_radius=window_radius,
        realign_window=realign_window,
        min_logfpr=min_logfpr,
    )
    score, shift, orientation, n_sites, metric_str = profile_compare(
        query_norm, query_anchors, target.bundle, target.anchors, config
    )
    return ComparisonResult(
        query.name, target.name, score, shift, orientation, metric_str, n_sites
    )
end
