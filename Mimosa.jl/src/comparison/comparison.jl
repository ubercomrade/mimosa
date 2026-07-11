# Motif comparison: metrics, alignment, results, and profile comparison.

include("metrics.jl")
include("alignment.jl")
include("profile_metrics.jl")
include("results.jl")
include("profile_comparison.jl")

export AbstractColumnMetric,
    PearsonCorrelation,
    EuclideanDistance,
    CosineSimilarity,
    metric_name,
    parse_metric,
    score_columns,
    MotifCandidate,
    Orientation,
    ORIENTATIONS,
    ComparisonResult,
    align_motif_matrices,
    score_motif_candidates,
    select_best,
    prepare_motif,
    compare,
    AbstractProfileMetric,
    OverlapCoefficient,
    OverlapCoefficientRowwise,
    DiceSimilarity,
    DiceSimilarityRowwise,
    CosineSimilarityProfile,
    parse_profile_metric,
    ProfileConfig,
    profile_compare
