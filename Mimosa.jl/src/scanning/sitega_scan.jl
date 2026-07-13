# SiteGA scanning adapter.
#
# All scanning logic (forward, reverse, best, both, single-seq, batch,
# in-place, result lengths) is handled by generic methods defined in
# `higher_order_scan.jl` for `AbstractHigherOrderMotif`. This file retains
# only the model-specific `npositions_sitega` function for backward-compatible
# dispatch and the public export.

"""
    npositions_sitega(seq_len::Int, model::SiteGA)

Return the number of scanning positions for a SiteGA model.
"""
function npositions_sitega(seq_len::Int, model::SiteGA)
    return npositions_ho(seq_len, model)
end
