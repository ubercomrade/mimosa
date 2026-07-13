# Profile comparison: metrics, results, and profile comparison.

include("profile_metrics.jl")
include("results.jl")
include("profile_comparison.jl")

export metric_name,
    ComparisonResult,
    compare,
    AbstractProfileMetric,
    OverlapCoefficient,
    OverlapCoefficientRowwise,
    DiceSimilarity,
    DiceSimilarityRowwise,
    CosineSimilarityProfile,
    parse_profile_metric,
    ProfileConfig,
    profile_compare,
    PreparedProfile,
    prepare_profile
