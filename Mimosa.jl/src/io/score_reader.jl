# Score profile reader: FASTA-like numerical score files.

"""
    read_scores(path::AbstractString)

Read a FASTA-like file of numerical score profiles into a [`ScoreProfile`](@ref).

Each `>` header starts a new profile. Subsequent lines contain
whitespace- or comma-separated float values. The model name is derived
from the filename (without extension).
"""
function read_scores(path::AbstractString)
    rows = Vector{Vector{Float32}}()
    current_values = Float32[]

    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        if startswith(stripped, '>')
            if !isempty(current_values)
                push!(rows, current_values)
                current_values = Float32[]
            end
        else
            # Replace commas with spaces and split
            cleaned = replace(stripped, ',' => ' ')
            for token in split(cleaned)
                if !isempty(token)
                    try
                        push!(current_values, parse(Float32, String(token)))
                    catch
                        throw(
                            ModelFormatError(String(path), "invalid score value: '$token'.")
                        )
                    end
                end
            end
        end
    end
    if !isempty(current_values)
        push!(rows, current_values)
    end

    name = splitext(basename(path))[1]
    return ScoreProfile(name, build_ragged(rows))
end
