# Portable model storage: TOML manifest + NPY binary blobs.
#
# Implements the v1 schema from ADR 0003. Models are saved as a directory
# bundle with a `manifest.toml` and binary numeric blobs in NPY format.
#
# Directory bundle structure:
#   path/
#   ├── manifest.toml
#   └── data/
#       └── weights.npy  (or frequencies.npy)
#
# The manifest contains:
#   - format magic and version
#   - model kind (pwm, pfm, bamm, sitega, dimont, slim)
#   - model name
#   - dtype, shape, layout
#   - background (for PWM)
#   - coordinate conventions
#   - provenance (tool versions, creation info)
#   - SHA-256 checksums for each blob
#
# Julia `Serialization` is never used as a user-facing format.

using TOML
using SHA

const MODEL_FORMAT_VERSION = 1

const MODEL_KINDS = Set(["pwm", "pfm", "bamm", "sitega", "dimont", "slim"])

# ── Model fingerprinting (for cache keys) ────────────────────────────────────

"""
    model_fingerprint(model::AbstractMotifModel)

Return a hex-encoded SHA-256 fingerprint of a model's content for cache
keying and null distribution compatibility tracking.
"""
function model_fingerprint(model::AbstractMotifModel)
    return content_fingerprint(model)
end

"""
    model_collection_fingerprint(models::AbstractVector{<:AbstractMotifModel})

Return a hex-encoded SHA-256 fingerprint of a collection of models,
incorporating each model's individual fingerprint in sorted order.
"""
function model_collection_fingerprint(models::AbstractVector{<:AbstractMotifModel})
    fps = sort!([model_fingerprint(m) for m in models])
    return content_fingerprint(join(fps, "|"))
end

# ── Save model ──────────────────────────────────────────────────────────────

"""
    writemodel(path, model; format=:auto)

Save a motif model to a portable directory bundle at `path`.

The bundle contains a `manifest.toml` with metadata and a `data/` directory
with NPY binary blobs. Writes are atomic (temp + rename).

Throws `ArgumentError` for unsupported model kinds.
"""
function writemodel(path::AbstractString, model::AbstractMotifModel; format::Symbol=:auto)
    data_dir = joinpath(path, "data")
    mkpath(data_dir)

    kind = _model_kind(model)
    arr = _model_array(model)
    arr_name = _model_array_name(model)

    # Write numeric data as NPY
    npy_path = joinpath(data_dir, "$(arr_name).npy")
    _write_npy_2d(npy_path, arr)
    checksum = _file_sha256(npy_path)

    # Build manifest
    manifest = Dict{String,Any}(
        "format" => "mimosa",
        "format_version" => MODEL_FORMAT_VERSION,
        "kind" => kind,
        "name" => model.name,
        "dtype" => "<f4",
        "shape" => [size(arr, 1), size(arr, 2)],
        "layout" => "row_major",
        "convention" => "axes: (base, position) for matrix models, (context_code, position) for higher-order",
        "provenance" => Dict{String,Any}("tool" => "Mimosa.jl", "version" => "0.1.0"),
        "arrays" => Dict{String,Any}(
            arr_name => Dict{String,Any}(
                "file" => "data/$(arr_name).npy",
                "dtype" => "<f4",
                "shape" => [size(arr, 1), size(arr, 2)],
                "checksum" => "sha256:$checksum",
            ),
        ),
    )

    # Model-specific fields
    if model isa PWM
        manifest["background"] = collect(model.background)
    elseif model isa BaMM
        manifest["order"] = model.order
        manifest["motif_length"] = _model_length(model)
    elseif model isa SiteGA
        manifest["motif_length"] = _model_length(model)
    elseif model isa AbstractHigherOrderMotif
        # Dimont, Slim: have `span` and `motif_length`
        manifest["span"] = model.span
        manifest["motif_length"] = _model_length(model)
    end

    # Write manifest atomically
    manifest_path = joinpath(path, "manifest.toml")
    tmp_path = manifest_path * ".tmp"
    open(tmp_path, "w") do io
        return TOML.print(io, manifest; sorted=true)
    end
    mv(tmp_path, manifest_path; force=true)

    return path
end

# ── Load model ──────────────────────────────────────────────────────────────

"""
    readmodel(path; format=:auto, kwargs...)

Read a motif model from a directory bundle (v1 format) or a legacy format
file. When `path` is a directory containing `manifest.toml`, the portable
bundle format is used. Otherwise, legacy format detection applies.
"""
function readmodel(
    path::AbstractString;
    format::Symbol=:auto,
    index::Integer=0,
    background::AbstractFloat=0.25f0,
    kwargs...,
)
    # Check for portable bundle
    manifest_path = joinpath(path, "manifest.toml")
    if isdir(path) && isfile(manifest_path)
        return _read_model_bundle(path, manifest_path)
    end

    # Legacy format detection
    fmt = format === :auto ? _detect_format(path) : format
    if fmt === :meme
        pfm = read_meme(path; index=index)
        return pwm_from_pfm(pfm; background=background)
    elseif fmt === :pfm
        pfm = read_pfm(path)
        return pwm_from_pfm(pfm; background=background)
    elseif fmt === :bamm
        order_val = get(kwargs, :order, nothing)
        return read_bamm(path; order=order_val)
    elseif fmt === :sitega
        return read_sitega(path)
    elseif fmt === :dimont
        return read_dimont(path)
    elseif fmt === :slim
        return read_slim(path)
    else
        throw(ModelFormatError(path, "unsupported format: $(fmt)."))
    end
end

function _read_model_bundle(path::AbstractString, manifest_path::AbstractString)
    manifest = TOML.parsefile(manifest_path)

    # Validate format
    fmt = get(manifest, "format", "")
    fmt != "mimosa" &&
        throw(ModelFormatError(path, "unknown format: '$fmt'. Expected 'mimosa'."))

    version = get(manifest, "format_version", 0)
    version > MODEL_FORMAT_VERSION && throw(
        ModelFormatError(
            path, "unsupported format version: $version (max $MODEL_FORMAT_VERSION)."
        ),
    )

    kind = get(manifest, "kind", "")
    kind in MODEL_KINDS || throw(ModelFormatError(path, "unknown model kind: '$kind'."))

    name = get(manifest, "name", "unknown")

    # Read arrays
    arrays = get(manifest, "arrays", Dict{String,Any}())

    if kind == "pwm"
        return _read_pwm_bundle(path, manifest, name)
    elseif kind == "pfm"
        return _read_pfm_bundle(path, manifest, name)
    elseif kind == "bamm"
        return _read_bamm_bundle(path, manifest, name)
    elseif kind == "sitega"
        return _read_sitega_bundle(path, manifest, name)
    elseif kind == "dimont"
        return _read_dimont_bundle(path, manifest, name)
    elseif kind == "slim"
        return _read_slim_bundle(path, manifest, name)
    end
end

function _validate_bundle_checksum(
    path::AbstractString, file_rel::AbstractString, expected_checksum::AbstractString
)
    file_path = joinpath(path, file_rel)
    isfile(file_path) || throw(ModelFormatError(path, "missing data file: $file_rel"))
    startswith(expected_checksum, "sha256:") ||
        throw(ModelFormatError(path, "invalid checksum format: $expected_checksum"))
    expected_hash = expected_checksum[8:end]
    actual_hash = _file_sha256(file_path)
    actual_hash != expected_hash &&
        throw(ModelFormatError(path, "checksum mismatch for $file_rel."))
    return file_path
end

function _read_pwm_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    arrays = get(manifest, "arrays", Dict{String,Any}())
    weights_info = get(arrays, "weights", nothing)
    weights_info === nothing &&
        throw(ModelFormatError(path, "PWM bundle missing 'weights' array."))
    weights_path = _validate_bundle_checksum(
        path, weights_info["file"], weights_info["checksum"]
    )
    weights = _read_npy_f32_2d(weights_path)
    bg_raw = get(manifest, "background", [0.25f0, 0.25f0, 0.25f0, 0.25f0])
    bg = NTuple{4,Float32}((
        Float32(bg_raw[1]), Float32(bg_raw[2]), Float32(bg_raw[3]), Float32(bg_raw[4])
    ))
    return PWM(name, weights, bg)
end

function _read_pfm_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    arrays = get(manifest, "arrays", Dict{String,Any}())
    freq_info = get(arrays, "frequencies", nothing)
    freq_info === nothing &&
        throw(ModelFormatError(path, "PFM bundle missing 'frequencies' array."))
    freq_path = _validate_bundle_checksum(path, freq_info["file"], freq_info["checksum"])
    frequencies = _read_npy_f32_2d(freq_path)
    return PFM(name, frequencies)
end

function _read_bamm_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    arrays = get(manifest, "arrays", Dict{String,Any}())
    rep_info = get(arrays, "representation", nothing)
    rep_info === nothing &&
        throw(ModelFormatError(path, "BaMM bundle missing 'representation' array."))
    rep_path = _validate_bundle_checksum(path, rep_info["file"], rep_info["checksum"])
    representation = _read_npy_f32_2d(rep_path)
    order = get(manifest, "order", 0)
    motif_length = get(manifest, "motif_length", size(representation, 2))
    return BaMM(name, representation, order, motif_length)
end

function _read_sitega_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    arrays = get(manifest, "arrays", Dict{String,Any}())
    rep_info = get(arrays, "representation", nothing)
    rep_info === nothing &&
        throw(ModelFormatError(path, "SiteGA bundle missing 'representation' array."))
    rep_path = _validate_bundle_checksum(path, rep_info["file"], rep_info["checksum"])
    representation = _read_npy_f32_2d(rep_path)
    motif_length = get(manifest, "motif_length", size(representation, 2))
    return SiteGA(name, representation, motif_length)
end

function _read_dimont_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    arrays = get(manifest, "arrays", Dict{String,Any}())
    rep_info = get(arrays, "representation", nothing)
    rep_info === nothing &&
        throw(ModelFormatError(path, "Dimont bundle missing 'representation' array."))
    rep_path = _validate_bundle_checksum(path, rep_info["file"], rep_info["checksum"])
    representation = _read_npy_f32_2d(rep_path)
    span = get(manifest, "span", 0)
    motif_length = get(manifest, "motif_length", size(representation, 2))
    return Dimont(name, representation, span, motif_length)
end

function _read_slim_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    arrays = get(manifest, "arrays", Dict{String,Any}())
    rep_info = get(arrays, "representation", nothing)
    rep_info === nothing &&
        throw(ModelFormatError(path, "Slim bundle missing 'representation' array."))
    rep_path = _validate_bundle_checksum(path, rep_info["file"], rep_info["checksum"])
    representation = _read_npy_f32_2d(rep_path)
    span = get(manifest, "span", 0)
    motif_length = get(manifest, "motif_length", size(representation, 2))
    return Slim(name, representation, span, motif_length)
end

# ── Model helpers ───────────────────────────────────────────────────────────

_model_kind(::PWM) = "pwm"
_model_kind(::PFM) = "pfm"
_model_kind(::BaMM) = "bamm"
_model_kind(::SiteGA) = "sitega"
_model_kind(::Dimont) = "dimont"
_model_kind(::Slim) = "slim"

_model_array(model::PWM) = model.weights
_model_array(model::PFM) = model.frequencies
_model_array(model::BaMM) = model.representation
_model_array(model::SiteGA) = model.representation
_model_array(model::Dimont) = model.representation
_model_array(model::Slim) = model.representation

_model_array_name(::PWM) = "weights"
_model_array_name(::PFM) = "frequencies"
_model_array_name(::AbstractHigherOrderMotif) = "representation"

_model_length(model::PWM) = length(model)
_model_length(model::PFM) = length(model)
_model_length(model::BaMM) = model.motif_length
_model_length(model::SiteGA) = model.motif_length
_model_length(model::Dimont) = model.motif_length
_model_length(model::Slim) = model.motif_length

# ── NPY writer (2D Float32) ──────────────────────────────────────────────────

function _write_npy_2d(path::AbstractString, data::AbstractMatrix{Float32})
    dtype = "<f4"
    nrows, ncols = size(data)
    shape_str = "($nrows, $ncols)"
    header_dict = "{'descr': '$dtype', 'fortran_order': False, 'shape': $shape_str, }"

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
        # Data (little-endian Float32, column-major = fortran_order=False for row-major data)
        # NPY stores data in C order (row-major). Julia is column-major.
        # We write in row-major order to match numpy convention.
        for r in 1:nrows
            for c in 1:ncols
                write(io, htol(data[r, c]))
            end
        end
    end
end

function _read_npy_f32_2d(path::AbstractString)
    open(path, "r") do io
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
        header = String(read(io, header_len))

        # Parse shape from header (simple parser for our own format)
        # Header looks like: {'descr': '<f4', 'fortran_order': False, 'shape': (5, 12), }
        shape_match = match(r"\'shape\':\s*\((\d+),\s*(\d+)\)", header)
        if shape_match === nothing
            error("Cannot parse NPY shape from header: $header")
        end
        nrows = parse(Int, shape_match.captures[1])
        ncols = parse(Int, shape_match.captures[2])

        # Read data (row-major)
        raw = read(io)
        expected_bytes = nrows * ncols * 4
        length(raw) < expected_bytes &&
            error("NPY data truncated: expected $expected_bytes bytes, got $(length(raw)).")

        data = Matrix{Float32}(undef, nrows, ncols)
        for r in 1:nrows
            for c in 1:ncols
                idx = ((r - 1) * ncols + (c - 1)) * 4 + 1
                data[r, c] = ltoh(reinterpret(Float32, raw[idx:(idx + 3)])[1])
            end
        end
        return data
    end
end

# ── SHA-256 file checksum ────────────────────────────────────────────────────

function _file_sha256(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(read(io)))
    end
end
