# Motif model I/O: MEME, PFM, score profile, BaMM, and SiteGA readers.

include("motif_readers.jl")
include("score_reader.jl")
include("bamm_reader.jl")
include("sitega_reader.jl")
include("dimont_reader.jl")
include("slim_reader.jl")
include("model_storage.jl")

export readmodel,
    read_meme,
    read_pfm,
    read_scores,
    read_bamm,
    read_sitega,
    read_dimont,
    read_slim,
    write_sitega,
    writemodel
