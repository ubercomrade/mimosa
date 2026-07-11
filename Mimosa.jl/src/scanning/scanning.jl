# Scanning module: strand policies, PWM scanning, and BaMM scanning kernels.

include("strands.jl")
include("pwm_scan.jl")
include("bamm_scan.jl")

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
    scan_result_lengths
