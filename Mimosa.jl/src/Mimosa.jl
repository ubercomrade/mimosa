"""
    Mimosa

A Julia package for motif scanning, comparison, and statistical evaluation.

Stage 1: PWM/PFM parsing, matrix metrics, motif alignment, and typed results.
Stage 2: Encoded sequence batches, FASTA reader, PWM scanning with strand
policies.

See `REFACTORING.md` and `PLAN.md` for the migration roadmap.
"""
module Mimosa

include("errors.jl")
include("parallel/parallel.jl")
include("models/models.jl")
include("sequences/sequences.jl")
include("models/score_profile.jl")
include("scanning/scanning.jl")
include("io/io.jl")
include("comparison/comparison.jl")
include("profiles/profiles.jl")
include("sites/sites.jl")
include("statistics/statistics.jl")
include("cache/cache.jl")
include("serialization.jl")
include("cli.jl")
include("precompile.jl")

export readmodel,
    read_meme,
    read_pfm,
    read_scores,
    read_bamm,
    read_sitega,
    read_dimont,
    read_slim,
    write_sitega,
    writemodel,
    readsequences,
    compare,
    to_json,
    to_dict,
    main
export MimosaError, ModelFormatError, ModelDimensionError, InvariantError

# Sequence / scanning exports
export EncodedSequenceBatch,
    nsequences,
    seqlength,
    sequence,
    empty_sequence_batch,
    make_random_sequences,
    encode_base,
    encode_sequence,
    reverse_complement,
    reverse_complement!,
    to_padded,
    from_padded,
    N_CODE
export RaggedArray, nrows, rowlength, row, build_ragged, empty_ragged
export StrandPolicy, ForwardOnly, ReverseOnly, BestStrand, BothStrands, StrandPair
export scan,
    scan!,
    scan_forward!,
    scan_reverse!,
    scan_best!,
    scan_both!,
    npositions,
    npositions_bamm,
    npositions_dimont,
    npositions_slim,
    scan_result_lengths

# Model exports
export ScoreProfile,
    BaMM, SiteGA, Dimont, Slim, scorebounds, profile_bundle, site_start_offset

# Sites and PFM reconstruction exports
export SiteSelector,
    BestPerSequence,
    ThresholdHits,
    TopFractionHits,
    SiteHit,
    SiteCollection,
    selectsites,
    reconstruct_pfm,
    extract_site_matrix,
    build_pcm,
    site_strings,
    sort_hits!,
    select_top_fraction,
    empty_site_collection

# Profile comparison exports
export AbstractProfileMetric,
    OverlapCoefficient,
    OverlapCoefficientRowwise,
    DiceSimilarity,
    DiceSimilarityRowwise,
    CosineSimilarityProfile,
    parse_profile_metric,
    ProfileConfig,
    profile_compare,
    PreparedProfile,
    prepare_profile,
    LogTailTable,
    EmpiricalLogTail,
    fit,
    lookup_score,
    transform_scores,
    flatten_bundle,
    normalize_bundle,
    AnchorCSR,
    build_anchor_csr,
    collect_best_anchors,
    collect_threshold_anchors,
    collect_anchors,
    score_shift

# Statistics exports
export GEVFit,
    GEVFitFailure,
    GEVFitResult,
    fit_gev,
    survival,
    cdf,
    scipy_params,
    BenjaminiHochberg,
    adjusted_pvalues,
    evalue,
    pvalue,
    GroupRelations,
    parse_group_relations,
    eligible_targets,
    NullStrategy,
    MotifNullStrategy,
    ProfileNullStrategy,
    NullDistribution,
    NullPair,
    NullBuildConfig,
    NullBuildResult,
    AnnotatedResult,
    build_null,
    annotate_results,
    ANNOTATED_RESULT_SCHEMA_VERSION,
    savenull,
    loadnull,
    NULL_FORMAT_VERSION

# Parallelism exports (Stage 7)
export ExecutionPolicy, SerialExecution, ThreadedExecution

# Cache exports (Stage 7)
export Cache,
    cache_key,
    cache_has,
    cache_get,
    cache_get_meta,
    cache_set,
    clearcache,
    content_fingerprint,
    model_fingerprint,
    model_collection_fingerprint,
    sequence_fingerprint,
    MODEL_FORMAT_VERSION

end # module
