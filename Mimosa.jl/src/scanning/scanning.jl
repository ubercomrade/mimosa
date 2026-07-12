# Scanning module: strand policies, PWM scanning, BaMM scanning, and SiteGA scanning kernels.

include("strands.jl")
include("pwm_scan.jl")
include("bamm_scan.jl")
include("sitega_scan.jl")
include("dimont_scan.jl")

export StrandPolicy,
    ForwardOnly,
    ReverseOnly,
    BestStrand,
    BothStrands,
    StrandPair,
    scan,
    scan!,
    scan_forward!,
    scan_reverse!,
    scan_best!,
    scan_both!,
    npositions,
    npositions_bamm,
    npositions_sitega,
    npositions_dimont,
    scan_result_lengths
