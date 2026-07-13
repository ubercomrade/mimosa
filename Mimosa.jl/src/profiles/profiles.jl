# Profile comparison sub-module: normalization, anchors, and alignment.

include("normalization.jl")
include("anchors.jl")
include("alignment.jl")

export LogTailTable,
    EmpiricalLogTail,
    fit,
    lookup_score,
    transform_scores,
    flatten_bundle,
    normalize_bundle,
    _fit_transform_empirical,
    AnchorCSR,
    build_anchor_csr,
    collect_best_anchors,
    collect_threshold_anchors,
    collect_anchors,
    score_shift,
    ProfileConfig,
    profile_compare,
    PreparedProfile,
    prepare_profile
