# Scanning module: strand policies and PWM scanning kernels.

include("strands.jl")
include("pwm_scan.jl")

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
