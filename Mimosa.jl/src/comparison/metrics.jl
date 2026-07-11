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

const SIMILARITY_EPS = Float32(1e-9)

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
    throw(ArgumentError("metric must be one of: 'pcc', 'ed', 'cosine', got '$name'."))
end

# Column-wise Pearson correlation of two `(base, overlap)` matrices.
function _column_pcc(x1::AbstractMatrix{T}, x2::AbstractMatrix{T}) where {T<:AbstractFloat}
    base_count = size(x1, 1)
    overlap = size(x1, 2)
    total = zero(T)
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
        total += denom > SIMILARITY_EPS ? num / denom : zero(T)
    end
    return total / T(overlap)
end

# Column-wise cosine similarity.
function _column_cosine(x1::AbstractMatrix{T}, x2::AbstractMatrix{T}) where {T<:AbstractFloat}
    base_count = size(x1, 1)
    overlap = size(x1, 2)
    total = zero(T)
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
        total += denom > SIMILARITY_EPS ? num / denom : zero(T)
    end
    return total / T(overlap)
end

# Column-wise Euclidean similarity = -mean distance.
function _column_euclidean(x1::AbstractMatrix{T}, x2::AbstractMatrix{T}) where {T<:AbstractFloat}
    base_count = size(x1, 1)
    overlap = size(x1, 2)
    total = zero(T)
    for col in 1:overlap
        dist_sq = zero(T)
        for b in 1:base_count
            d = x1[b, col] - x2[b, col]
            dist_sq += d * d
        end
        total -= sqrt(dist_sq)
    end
    return total / T(overlap)
end

"""
    score_columns(metric, query_columns, target_columns)

Score one aligned column block using the given metric. Returns the aggregated
similarity (higher is better) matching Python's `_score_motif_columns`.
"""
function score_columns(metric::AbstractColumnMetric, query::AbstractMatrix{T}, target::AbstractMatrix{T}) where {T<:AbstractFloat}
    if size(query) != size(target)
        throw(ModelDimensionError("column block shape mismatch: $(size(query)) vs $(size(target))."))
    end
    metric isa PearsonCorrelation && return _column_pcc(query, target)
    metric isa CosineSimilarity && return _column_cosine(query, target)
    metric isa EuclideanDistance && return _column_euclidean(query, target)
    throw(ArgumentError("unsupported metric: $(metric)."))
end