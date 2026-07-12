# Portable null-distribution storage: TOML manifest + NPY binary blobs.
#
# Implements the v1 schema from ADR 0003. The manifest is a TOML file
# (stdlib, no external dependency) with metadata and array references.
# Binary data uses the standard NPY format (little-endian Float64).
#
# Directory bundle structure:
#   path/
#   ├── manifest.toml
#   └── data/
#       └── raw_null_scores.npy

using TOML
using SHA

const NULL_FORMAT_VERSION = 1

"""
    savenull(path, dist::NullDistribution)

Save a null distribution to a portable directory bundle at `path`.

Creates a `manifest.toml` and a `data/raw_null_scores.npy` file.
Uses atomic write (temp file + rename) for safety.
"""
function savenull(path::AbstractString, dist::NullDistribution)
    # Create directory structure
    data_dir = joinpath(path, "data")
    mkpath(data_dir)

    # Write raw scores as NPY
    npy_path = joinpath(data_dir, "raw_null_scores.npy")
    _write_npy(npy_path, dist.raw_scores)
    checksum = _file_sha256(npy_path)

    # Build manifest
    fit_result = dist.fit
    if fit_result isa GEVFit
        gev_params = [fit_result.shape, fit_result.location, fit_result.scale]
        estimator_type = "genextreme"
        converged = fit_result.converged
        iterations = fit_result.iterations
        loglikelihood = fit_result.loglikelihood
    else
        gev_params = [0.0, 0.0, 0.0]
        estimator_type = "failed"
        converged = false
        iterations = fit_result.iterations
        loglikelihood = 0.0
    end

    manifest = Dict{String,Any}(
        "format" => "mimosa",
        "format_version" => NULL_FORMAT_VERSION,
        "kind" => "null_distribution",
        "strategy" => dist.strategy,
        "metric" => dist.metric,
        "estimator_type" => estimator_type,
        "genextreme_params" => gev_params,
        "genextreme_converged" => converged,
        "genextreme_iterations" => iterations,
        "genextreme_loglikelihood" => loglikelihood,
        "n_null" => dist.n_null,
        "n_queries" => dist.n_queries,
        "skipped" => [Dict("query" => s.query, "reason" => s.reason) for s in dist.skipped],
        "compatibility" => Dict{String,Any}(
            "format_version" => NULL_FORMAT_VERSION,
            "strategy" => dist.strategy,
            "metric" => dist.metric,
            "sequence_fingerprint" => dist.sequence_fingerprint,
            "background_fingerprint" => dist.background_fingerprint,
            "model_collection_fingerprint" =>
                if dist.model_collection_fingerprint === nothing
                    "none"
                else
                    dist.model_collection_fingerprint
                end,
            "relation_fingerprint" => if dist.relation_fingerprint === nothing
                "none"
            else
                dist.relation_fingerprint
            end,
        ),
        "arrays" => Dict{String,Any}(
            "raw_null_scores" => Dict{String,Any}(
                "file" => "data/raw_null_scores.npy",
                "dtype" => "<f8",
                "shape" => [length(dist.raw_scores)],
                "checksum" => "sha256:$checksum",
            ),
        ),
    )

    # Write manifest atomically
    manifest_path = joinpath(path, "manifest.toml")
    tmp_path = manifest_path * ".tmp"
    open(tmp_path, "w") do io
        return TOML.print(io, manifest; sorted=true)
    end
    mv(tmp_path, manifest_path; force=true)

    return path
end

"""
    loadnull(path)

Load a null distribution from a directory bundle at `path`.

Validates the manifest format version and checksums. Returns a
[`NullDistribution`](@ref).
"""
function loadnull(path::AbstractString)
    manifest_path = joinpath(path, "manifest.toml")
    if !isfile(manifest_path)
        throw(ArgumentError("No manifest.toml found at: $path"))
    end

    manifest = TOML.parsefile(manifest_path)

    # Validate format
    format = get(manifest, "format", "")
    if format != "mimosa"
        throw(ArgumentError("Unknown format: '$format'. Expected 'mimosa'."))
    end

    version = get(manifest, "format_version", 0)
    if version > NULL_FORMAT_VERSION
        throw(
            ArgumentError(
                "Unsupported format version: $version (max $NULL_FORMAT_VERSION)."
            ),
        )
    end

    kind = get(manifest, "kind", "")
    if kind != "null_distribution"
        throw(ArgumentError("Expected kind 'null_distribution', got '$kind'."))
    end

    strategy = get(manifest, "strategy", "")
    metric = get(manifest, "metric", "")

    # Read raw scores
    arrays = get(manifest, "arrays", Dict{String,Any}())
    raw_info = get(arrays, "raw_null_scores", nothing)
    if raw_info === nothing
        throw(ArgumentError("Null distribution is missing raw_null_scores."))
    end

    npy_file = get(raw_info, "file", "")
    npy_path = joinpath(path, npy_file)
    if !isfile(npy_path)
        throw(ArgumentError("Null scores file not found: $npy_path"))
    end

    # Validate checksum
    expected_checksum = get(raw_info, "checksum", "")
    if startswith(expected_checksum, "sha256:")
        expected_hash = expected_checksum[8:end]
        actual_hash = _file_sha256(npy_path)
        if actual_hash != expected_hash
            throw(ArgumentError("Checksum mismatch for $npy_path."))
        end
    end

    raw_scores = _read_npy_f64(npy_path)

    # Reconstruct GEV fit
    gev_params = get(manifest, "genextreme_params", [0.0, 0.0, 0.0])
    estimator_type = get(manifest, "estimator_type", "")
    converged = get(manifest, "genextreme_converged", false)
    iterations = get(manifest, "genextreme_iterations", 0)
    loglikelihood = get(manifest, "genextreme_loglikelihood", 0.0)

    if estimator_type == "genextreme" && length(gev_params) == 3
        fit_result = GEVFit(
            gev_params[1],
            gev_params[2],
            gev_params[3],
            converged,
            iterations,
            loglikelihood,
        )
    else
        fit_result = GEVFitFailure(
            "Stored null distribution has a failed GEV fit.", length(raw_scores), iterations
        )
    end

    n_null = get(manifest, "n_null", length(raw_scores))
    n_queries = get(manifest, "n_queries", 0)

    # Read skipped
    skipped_raw = get(manifest, "skipped", [])
    skipped = [
        NamedTuple{(:query, :reason),Tuple{String,String}}((
            String(s["query"]), String(s["reason"])
        )) for s in skipped_raw
    ]

    # Read compatibility metadata
    compat = get(manifest, "compatibility", Dict{String,Any}())
    seq_fp = get(compat, "sequence_fingerprint", "none")
    bg_fp = get(compat, "background_fingerprint", "none")
    mcf = get(compat, "model_collection_fingerprint", nothing)
    rf = get(compat, "relation_fingerprint", nothing)

    return NullDistribution(
        strategy,
        metric,
        fit_result,
        raw_scores,
        NullPair[],  # pairs not stored in manifest (only raw scores)
        n_null,
        n_queries,
        skipped,
        mcf isa AbstractString && mcf != "none" ? String(mcf) : nothing,
        rf isa AbstractString && rf != "none" ? String(rf) : nothing,
        String(seq_fp),
        String(bg_fp),
    )
end

# ---------------------------------------------------------------------------
# NPY writer (minimal: 1D Float64)
# ---------------------------------------------------------------------------

function _write_npy(path::AbstractString, data::Vector{Float64})
    dtype = "<f8"
    shape_str = "($(length(data)),)"
    header_dict = "{'descr': '$dtype', 'fortran_order': False, 'shape': $shape_str, }"

    # Pad header so that (6 + 2 + 2 + len(header) + 1) is divisible by 64
    total = 10 + length(header_dict) + 1
    padding = (64 - total % 64) % 64
    header = header_dict * repeat(" ", padding) * "\n"

    open(path, "w") do io
        # Magic: \x93NUMPY
        write(io, UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])
        # Version 1.0
        write(io, UInt8(1), UInt8(0))
        # Header length (UInt16 LE)
        write(io, UInt16(length(header)))
        # Header
        write(io, header)
        # Data (little-endian Float64)
        for x in data
            write(io, htol(x))
        end
    end
end

function _read_npy_f64(path::AbstractString)
    open(path, "r") do io
        # Read magic
        magic = read(io, 6)
        if magic != UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]
            error("Not a valid NPY file: $path")
        end
        major = read(io, UInt8)
        minor = read(io, UInt8)
        if major == 1
            header_len = read(io, UInt16)
        elseif major == 2
            header_len = read(io, UInt32)
        else
            error("Unsupported NPY version: $major.$minor")
        end
        # Skip header (we know it's float64 1D from our own writer)
        read(io, header_len)
        # Read data
        raw = read(io)
        n = div(length(raw), 8)
        data = Vector{Float64}(undef, n)
        for i in 1:n
            bytes = raw[((i - 1) * 8 + 1):(i * 8)]
            data[i] = ltoh(reinterpret(Float64, bytes)[1])
        end
        return data
    end
end

# ---------------------------------------------------------------------------
# SHA-256 file checksum
# ---------------------------------------------------------------------------

function _file_sha256(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(read(io)))
    end
end
