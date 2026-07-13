# BaMM scanning adapter.
#
# All scanning logic (forward, reverse, best, both, single-seq, batch,
# in-place, result lengths) is handled by generic methods defined in
# `higher_order_scan.jl` for `AbstractHigherOrderMotif`. This file retains
# only the model-specific `npositions_bamm` function for backward-compatible
# dispatch and the public export.

"""
    npositions_bamm(seq_len::Int, model::BaMM)

Return the number of scanning positions for a BaMM model.
"""
function npositions_bamm(seq_len::Int, model::BaMM)
    return npositions_ho(seq_len, model)
end
