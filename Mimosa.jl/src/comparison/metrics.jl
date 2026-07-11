# Column metrics for motif matrix comparison.

abstract type AbstractColumnMetric end

"""
    PearsonCorrelation

Column-wise Pearson correlation coefficient, averaged over the overlap.
Higher is better. Zero-variance columns contribute 0.
"""
struct PearsonCorrelation <: AbstractColumnMetric end

"""
    EuclideanDistance

Column-wise Euclidean similarity: negative mean of per-column distances.
Higher is better (0 for identical columns).
"""
struct EuclideanDistance <: AbstractColumnMetric end

"""
    CosineSimilarity

Column-wise cosine similarity, averaged over the overlap.
Higher is better. Zero-norm columns contribute 0.
"""
struct CosineSimilarity <: AbstractColumnMetric end

const SIMILARITY_EPS_F32 = Float32(1e-9)
const SIMILARITY_EPS_F64 = Float64(1e-9)

"""
    metric_name(::AbstractColumnMetric)

Return the canonical string identifier for a metric, matching Python's
`metric` field in `ComparisonResult`.
"""
metric_name(::PearsonCorrelation) = "pcc"
metric_name(::EuclideanDistance) = "ed"
metric_name(::CosineSimilarity) = "cosine"

"""
    parse_metric(name)

Convert a metric string (`pcc`, `ed`, `cosine`) to a typed metric value.
"""
function parse_metric(name::AbstractString)
    name == "pcc" && return PearsonCorrelation()
    name == "ed" && return EuclideanDistance()
    name == "cosine" && return CosineSimilarity()
    return throw(
        ArgumentError("metric must be one of: 'pcc', 'ed', 'cosine', got '$name'.")
    )
end

# Column-wise Pearson correlation of two `(base, overlap)` matrices.
#
# Per-column PCC is computed in Float32 (matching Python's `_vectorized_pcc`),
# then clamped to [-1, 1] since PCC is mathematically bounded and values like
# 1.0000001 are Float32 artifacts from `sqrt(x)*sqrt(x) != x`. The sum across
# columns uses Float64 accumulation to match NumPy's `np.sum` for float32
# arrays. The final result is T(Float32(Float64_sum)) / T(overlap).
function _column_pcc(x1::AbstractMatrix{T}, x2::AbstractMatrix{T}) where {T<:AbstractFloat}
    base_count = size(x1, 1)
    overlap = size(x1, 2)
    total = zero(Float64)
    for col in 1:overlap
        mean1 = zero(T)
        mean2 = zero(T)
        for b in 1:base_count
            mean1 += x1[b, col]
            mean2 += x2[b, col]
        end
        mean1 /= T(base_count)
        mean2 /= T(base_count)
        num = zero(T)
        d1 = zero(T)
        d2 = zero(T)
        for b in 1:base_count
            c1 = x1[b, col] - mean1
            c2 = x2[b, col] - mean2
            num += c1 * c2
            d1 += c1 * c1
            d2 += c2 * c2
        end
        denom = sqrt(d1) * sqrt(d2)
        if denom > SIMILARITY_EPS_F32
            pcc_val = num / denom
            # Clamp to [-1, 1]: PCC is mathematically bounded, and Float32
            # sqrt(x)*sqrt(x) can give values like 1.0000001.
            pcc_val = clamp(pcc_val, T(-1), T(1))
        else
            pcc_val = zero(T)
        end
        total += Float64(pcc_val)
    end
    return T(Float32(total)) / T(overlap)
end

# Column-wise cosine similarity.
# Per-column cosine is computed in Float32 (matching Python's
# `_vectorized_cosine`), summed with Float64 accumulation.
function _column_cosine(
    x1::AbstractMatrix{T}, x2::AbstractMatrix{T}
) where {T<:AbstractFloat}
    base_count = size(x1, 1)
    overlap = size(x1, 2)
    total = zero(Float64)
    for col in 1:overlap
        num = zero(T)
        n1 = zero(T)
        n2 = zero(T)
        for b in 1:base_count
            num += x1[b, col] * x2[b, col]
            n1 += x1[b, col] * x1[b, col]
            n2 += x2[b, col] * x2[b, col]
        end
        denom = sqrt(n1) * sqrt(n2)
        if denom > SIMILARITY_EPS_F32
            cos_val = num / denom
            # Clamp to [-1, 1]: cosine similarity is mathematically bounded.
            cos_val = clamp(cos_val, T(-1), T(1))
        else
            cos_val = zero(T)
        end
        total += Float64(cos_val)
    end
    return T(Float32(total)) / T(overlap)
end

# Column-wise Euclidean similarity = -mean distance.
# Per-column distance is computed in Float32, summed with Float64 accumulation.
function _column_euclidean(
    x1::AbstractMatrix{T}, x2::AbstractMatrix{T}
) where {T<:AbstractFloat}
    base_count = size(x1, 1)
    overlap = size(x1, 2)
    total = zero(Float64)
    for col in 1:overlap
        dist_sq = zero(T)
        for b in 1:base_count
            d = x1[b, col] - x2[b, col]
            dist_sq += d * d
        end
        dist_val = sqrt(dist_sq)
        total -= Float64(dist_val)
    end
    return T(Float32(total)) / T(overlap)
end

"""
    score_columns(metric, query_columns, target_columns)

Score one aligned column block using the given metric. Returns the aggregated
similarity (higher is better) matching Python's `_score_motif_columns`.
"""
function score_columns(
    metric::AbstractColumnMetric, query::AbstractMatrix{T}, target::AbstractMatrix{T}
) where {T<:AbstractFloat}
    if size(query) != size(target)
        throw(
            ModelDimensionError(
                "column block shape mismatch: $(size(query)) vs $(size(target))."
            ),
        )
    end
    metric isa PearsonCorrelation && return _column_pcc(query, target)
    metric isa CosineSimilarity && return _column_cosine(query, target)
    metric isa EuclideanDistance && return _column_euclidean(query, target)
    return throw(ArgumentError("unsupported metric: $(metric)."))
end
