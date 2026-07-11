# Thin CLI for motif comparison (Stage 1 demonstration slice).

# This module is a thin adapter: argument parsing → typed config → public API
# → JSON serialization. No scanning, metrics, or alignment logic lives here.

# Uses symbols defined in the enclosing Mimosa module: readmodel, compare,
# to_json, parse_metric, PWM, PFM.

function _print_usage(io::IO)
    println(
        io, "usage: mimosa motif --query <path> --target <path> [--metric pcc|ed|cosine]"
    )
    return println(io, "       [--background 0.25] [--query-index 0] [--target-index 0]")
end

function _parse_args(args::Vector{String})
    kwargs = Dict{Symbol,Any}(
        :metric => "pcc", :background => 0.25, :query_index => 0, :target_index => 0
    )
    i = 1
    query = nothing
    target = nothing
    while i <= length(args)
        arg = args[i]
        if arg == "--query"
            i += 1
            query = args[i]
        elseif arg == "--target"
            i += 1
            target = args[i]
        elseif arg == "--metric"
            i += 1
            kwargs[:metric] = args[i]
        elseif arg == "--background"
            i += 1
            kwargs[:background] = parse(Float32, args[i])
        elseif arg == "--query-index"
            i += 1
            kwargs[:query_index] = parse(Int, args[i])
        elseif arg == "--target-index"
            i += 1
            kwargs[:target_index] = parse(Int, args[i])
        elseif arg in ("--help", "-h")
            _print_usage(stdout)
            return nothing
        else
            println(stderr, "unknown argument: $arg")
            _print_usage(stderr)
            return nothing
        end
        i += 1
    end
    if query === nothing || target === nothing
        println(stderr, "error: --query and --target are required.")
        _print_usage(stderr)
        return nothing
    end
    return (query, target, kwargs)
end

"""
    main(args=ARGS)

Thin CLI entry point for motif comparison. Returns an integer exit code (0 on
success, 1 on error). Prints JSON to stdout on success, errors to stderr.
"""
function main(args::Vector{String}=ARGS)::Int
    parsed = _parse_args(args)
    parsed === nothing && return 1
    query_path, target_path, kwargs = parsed
    try
        query_model = readmodel(
            query_path; index=kwargs[:query_index], background=kwargs[:background]
        )
        target_model = readmodel(
            target_path; index=kwargs[:target_index], background=kwargs[:background]
        )
        result = compare(query_model, target_model; metric=kwargs[:metric])
        println(to_json(result))
        return 0
    catch e
        println(stderr, "error: $(typeof(e).name.name): $(e)")
        return 1
    end
end
