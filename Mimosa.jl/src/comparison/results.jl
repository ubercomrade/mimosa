# Comparison result type and public `compare` entry point for motif matrices.

"""
    ComparisonResult

Immutable result of a motif comparison.

Fields:
- `query::String`: query model name.
- `target::String`: target model name.
- `score::Float32`: best alignment score (higher is better).
- `offset::Int`: offset of the query relative to the target at the best
  alignment. Positive means the query is shifted right.
- `orientation::String`: one of `"++"`, `"+-"`, `"-+"`, `"--"`.
- `metric::String`: canonical metric identifier (`pcc`, `ed`, `cosine`).
"""
struct ComparisonResult
    query::String
    target::String
    score::Float32
    offset::Int
    orientation::String
    metric::String
end

"""
    compare(query::PWM, target::PWM; metric=:pcc)

Compare two [`PWM`](@ref) models by direct matrix alignment across all offsets
and four orientations, returning the best [`ComparisonResult`](@ref) with
deterministic tie-breaking per ADR 0006.
"""
function compare(
    query::PWM, target::PWM; metric::Union{AbstractString,Symbol,AbstractColumnMetric}=:pcc
)
    m = _resolve_metric(metric)
    q_fwd, q_rev = prepare_motif(query.weights)
    t_fwd, t_rev = prepare_motif(target.weights)
    candidates = score_motif_candidates(q_fwd, q_rev, t_fwd, t_rev, m)
    best = select_best(candidates)
    return ComparisonResult(
        query.name,
        target.name,
        best.score,
        best.offset,
        best.orientation.label,
        metric_name(m),
    )
end

function compare(query::PWM, target::PFM; kwargs...)
    return throw(
        ArgumentError(
            "PWM vs PFM direct comparison is not supported at Stage 1; convert PFM to PWM first.",
        ),
    )
end

function compare(
    query::PFM, target::PFM; metric::Union{AbstractString,Symbol,AbstractColumnMetric}=:pcc
)
    m = _resolve_metric(metric)
    q_fwd = query.frequencies
    q_rev = reverse_complement(q_fwd)
    t_fwd = target.frequencies
    t_rev = reverse_complement(t_fwd)
    candidates = score_motif_candidates(q_fwd, q_rev, t_fwd, t_rev, m)
    best = select_best(candidates)
    return ComparisonResult(
        query.name,
        target.name,
        best.score,
        best.offset,
        best.orientation.label,
        metric_name(m),
    )
end

function _resolve_metric(metric)
    metric isa AbstractColumnMetric && return metric
    metric isa Symbol && return parse_metric(string(metric))
    return parse_metric(metric)
end
