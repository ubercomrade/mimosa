# Profile comparison dispatch: compare ScoreProfile models using the profile algorithm.

"""
    compare(query::ScoreProfile, target::ScoreProfile; metric=:co, kwargs...)

Compare two [`ScoreProfile`](@ref) models using the window-based profile
comparison algorithm. Returns a [`ComparisonResult`](@ref) with deterministic
tie-breaking per ADR 0006.

Keyword arguments:
- `metric`: profile metric (`:co`, `:co_rowwise`, `:dice`, `:dice_rowwise`,
  `:cosine`, or a typed `AbstractProfileMetric`). Default `:co`.
- `search_range::Int=10`: maximum shift to search.
- `window_radius::Int=10`: half-window size for site windows.
- `realign_window::Int=3`: realignment search radius for target anchors.
- `min_logfpr::Float32=0.0`: minimum log FPR for threshold anchors (0 = best anchors).

The comparison pipeline:
1. Resolve profile bundles (both strands = same scores for ScoreProfile).
2. Fit `EmpiricalLogTail` normalization from each model's own scores.
3. Apply normalization to both strands.
4. Collect anchors (best per row or threshold).
5. Score all four orientation pairs across all shifts.
6. Select best with deterministic tie-breaking.
"""
function compare(
    query::ScoreProfile,
    target::ScoreProfile;
    metric::Union{AbstractString,Symbol,AbstractProfileMetric}=:co,
    search_range::Int=10,
    window_radius::Int=10,
    realign_window::Int=3,
    min_logfpr::Float32=Float32(0.0),
)
    m = _resolve_profile_metric(metric)

    # Resolve profile bundles (both strands = same for ScoreProfile)
    query_raw = profile_bundle(query)
    target_raw = profile_bundle(target)

    # Fit normalization from each model's own scores
    query_flat = flatten_bundle(query_raw)
    target_flat = flatten_bundle(target_raw)

    query_table = fit(EmpiricalLogTail(), query_flat)
    target_table = fit(EmpiricalLogTail(), target_flat)

    # Apply normalization
    query_norm = normalize_bundle(query_table, query_raw)
    target_norm = normalize_bundle(target_table, target_raw)

    # Configure and compare
    config = ProfileConfig(;
        metric=m,
        search_range=search_range,
        window_radius=window_radius,
        realign_window=realign_window,
        min_logfpr=min_logfpr,
    )

    score, shift, orientation, n_sites, metric_str = profile_compare(
        query_norm, target_norm, config
    )

    return ComparisonResult(
        query.name, target.name, score, shift, orientation, metric_str, n_sites
    )
end
