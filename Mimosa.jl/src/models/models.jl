# Model type hierarchy and matrix conversions for Mimosa.

include("types.jl")
include("matrices.jl")

export AbstractMotifModel,
    AbstractMatrixMotif,
    AbstractHigherOrderMotif,
    PFM,
    PWM,
    pcm_to_pfm,
    pfm_to_pwm,
    pwm_from_pfm,
    extend_pwm_with_n,
    reverse_complement,
    scorebounds
