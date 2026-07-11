# Motif comparison: metrics, alignment, and results.

include("metrics.jl")
include("alignment.jl")
include("results.jl")

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
    compare
