# Dimont scanning adapter.
#
# All scanning logic (forward, reverse, best, both, single-seq, batch,
# in-place, result lengths) is handled by generic methods defined in
# `higher_order_scan.jl` for `AbstractHigherOrderMotif`. This file retains
# only the model-specific `npositions_dimont` function for backward-compatible
# dispatch and the public export.

"""
    npositions_dimont(seq_len::Int, model::Dimont)

Return the number of scanning positions for a Dimont model.
"""
function npositions_dimont(seq_len::Int, model::Dimont)
    return npositions_ho(seq_len, model)
end
