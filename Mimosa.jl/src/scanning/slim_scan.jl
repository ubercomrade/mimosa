# Slim scanning adapter.
#
# All scanning logic (forward, reverse, best, both, single-seq, batch,
# in-place, result lengths) is handled by generic methods defined in
# `higher_order_scan.jl` for `AbstractHigherOrderMotif`. This file retains
# only the model-specific `npositions_slim` function for backward-compatible
# dispatch and the public export.

"""
    npositions_slim(seq_len::Int, model::Slim)

Return the number of scanning positions for a Slim model.
"""
function npositions_slim(seq_len::Int, model::Slim)
    return npositions_ho(seq_len, model)
end
