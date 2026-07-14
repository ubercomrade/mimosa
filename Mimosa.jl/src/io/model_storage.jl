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

"""
    MODEL_FORMAT_VERSION

Current version of the portable model bundle format.
Value is `1`. Bundles with a different version are rejected on load.
"""
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
with NPY binary blobs. Writes use a complete sibling staging directory and
atomic rename.

Raises `InvariantError` when a model cannot be represented by the v1 bundle.
"""
function writemodel(path::AbstractString, model::AbstractMotifModel; format::Symbol=:auto)
    format in (:auto, :bundle) ||
        throw(ArgumentError("unsupported model output format '$format'."))
    kind = _model_kind(model)
    arr = _model_array(model)
    arr_name = _model_array_name(model)
    _bundle_shape_payload_bytes(
        [size(arr, 1), size(arr, 2)], "<f4", String(path), "model array"
    )
    all(isfinite, arr) || throw(InvariantError("model array contains non-finite values."))

    return _with_bundle_write(path) do target, stage
        data_dir = joinpath(stage, BUNDLE_DATA_DIR)
        npy_path = joinpath(data_dir, "$(arr_name).npy")
        _write_npy_2d(npy_path, arr)
        checksum = _file_sha256(npy_path)
        shape = [size(arr, 1), size(arr, 2)]

        manifest = Dict{String,Any}(
            "format" => "mimosa",
            "format_version" => MODEL_FORMAT_VERSION,
            "kind" => kind,
            "name" => model.name,
            "dtype" => "<f4",
            "shape" => shape,
            "layout" => "row_major",
            "convention" => "axes: (base, position) for matrix models, (context_code, position) for higher-order",
            "provenance" => Dict{String,Any}(
                "tool" => "Mimosa.jl", "version" => string(Base.pkgversion(@__MODULE__))
            ),
            "arrays" => Dict{String,Any}(
                arr_name => Dict{String,Any}(
                    "file" => "data/$(arr_name).npy",
                    "dtype" => "<f4",
                    "shape" => shape,
                    "checksum" => "sha256:$checksum",
                ),
            ),
        )

        if model isa PWM
            manifest["background"] = collect(model.background)
        elseif model isa BaMM
            manifest["order"] = model.order
            manifest["motif_length"] = _model_length(model)
        elseif model isa SiteGA
            manifest["motif_length"] = _model_length(model)
        elseif model isa AbstractHigherOrderMotif
            manifest["span"] = model.span
            manifest["motif_length"] = _model_length(model)
        end

        _write_bundle_manifest(joinpath(stage, BUNDLE_MANIFEST_NAME), manifest)
        return target
    end
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
    elseif isdir(path)
        throw(ModelFormatError(path, "bundle is missing manifest.toml."))
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
    try
        manifest = _read_bundle_manifest(
            path, MODEL_FORMAT_VERSION; manifest_path=manifest_path
        )
        kind = _required_manifest_string(manifest, "kind", path, "manifest")
        kind in MODEL_KINDS || throw(_bundle_error(path, "unknown model kind '$kind'."))
        name = _required_manifest_string(manifest, "name", path, "manifest")

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
        else
            return _read_slim_bundle(path, manifest, name)
        end
    catch err
        _rethrow_bundle_error(path, err)
    end
end

function _read_model_array(
    path::AbstractString, manifest::AbstractDict, array_name::AbstractString
)
    layout = _required_manifest_string(manifest, "layout", path, "manifest")
    layout == "row_major" ||
        throw(_bundle_error(path, "unsupported array layout '$layout'."))
    dtype = _required_manifest_string(manifest, "dtype", path, "manifest")
    shape_value = haskey(manifest, "shape") ? manifest["shape"] : nothing
    shape_value === nothing && throw(_bundle_error(path, "manifest is missing 'shape'."))
    shape = _required_manifest_shape(shape_value, path, "manifest")
    spec = _parse_bundle_array(
        path,
        _required_manifest_table(manifest, "arrays", path, "manifest"),
        array_name,
        path,
    )
    spec.dtype == dtype ||
        throw(_bundle_error(path, "manifest and array dtype declarations disagree."))
    spec.shape == shape ||
        throw(_bundle_error(path, "manifest and array shape declarations disagree."))
    dtype == "<f4" || throw(_bundle_error(path, "model arrays must use dtype '<f4'."))
    file_path = _validate_bundle_array_checksum(path, spec, path)
    data = _read_npy_f32_2d(file_path; expected_shape=shape)
    all(isfinite, data) ||
        throw(_bundle_error(path, "model array contains non-finite values."))
    return data
end

function _validate_declared_model_shape(
    path::AbstractString,
    manifest::AbstractDict,
    expected_rows::Int,
    expected_columns::Int,
    model_kind::AbstractString,
)
    shape_value = haskey(manifest, "shape") ? manifest["shape"] : nothing
    shape_value === nothing && throw(_bundle_error(path, "manifest is missing 'shape'."))
    shape = _required_manifest_shape(shape_value, path, "manifest")
    shape == [expected_rows, expected_columns] || throw(
        _bundle_error(
            path,
            "$model_kind manifest shape does not match model constructor invariants.",
        ),
    )
    return shape
end

function _read_pwm_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    shape_value = haskey(manifest, "shape") ? manifest["shape"] : nothing
    shape_value === nothing && throw(_bundle_error(path, "manifest is missing 'shape'."))
    declared_shape = _required_manifest_shape(shape_value, path, "manifest")
    length(declared_shape) == 2 && declared_shape[1] == 5 && declared_shape[2] > 0 ||
        throw(_bundle_error(path, "PWM manifest shape must be [5, positive motif_length]."))
    _validate_declared_model_shape(path, manifest, 5, declared_shape[2], "PWM")
    weights = _read_model_array(path, manifest, "weights")
    bg_values = _required_manifest_floats(
        manifest, "background", path, "PWM manifest"; expected_length=4
    )
    bg = ntuple(i -> Float32(bg_values[i]), 4)
    all(isfinite, bg) ||
        throw(_bundle_error(path, "PWM background is not representable as Float32."))
    return PWM(name, weights, bg)
end

function _read_pfm_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    shape_value = haskey(manifest, "shape") ? manifest["shape"] : nothing
    shape_value === nothing && throw(_bundle_error(path, "manifest is missing 'shape'."))
    declared_shape = _required_manifest_shape(shape_value, path, "manifest")
    length(declared_shape) == 2 && declared_shape[1] == 4 && declared_shape[2] > 0 ||
        throw(_bundle_error(path, "PFM manifest shape must be [4, positive motif_length]."))
    _validate_declared_model_shape(path, manifest, 4, declared_shape[2], "PFM")
    frequencies = _read_model_array(path, manifest, "frequencies")
    return PFM(name, frequencies)
end

function _read_bamm_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    order = _required_manifest_int(
        manifest, "order", path, "BaMM manifest"; minimum=0, maximum=MAX_BAMM_ORDER
    )
    motif_length = _required_manifest_int(
        manifest,
        "motif_length",
        path,
        "BaMM manifest";
        minimum=1,
        maximum=MAX_BAMM_POSITIONS,
    )
    _validate_declared_model_shape(path, manifest, 5^(order + 1), motif_length, "BaMM")
    representation = _read_model_array(path, manifest, "representation")
    return BaMM(name, representation, order, motif_length)
end

function _read_sitega_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    motif_length = _required_manifest_int(
        manifest,
        "motif_length",
        path,
        "SiteGA manifest";
        minimum=1,
        maximum=MAX_SITEGA_LENGTH,
    )
    _validate_declared_model_shape(path, manifest, 25, motif_length, "SiteGA")
    representation = _read_model_array(path, manifest, "representation")
    return SiteGA(name, representation, motif_length)
end

function _read_dimont_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    span = _required_manifest_int(
        manifest, "span", path, "Dimont manifest"; minimum=0, maximum=DIMONT_MAX_SPAN
    )
    motif_length = _required_manifest_int(
        manifest,
        "motif_length",
        path,
        "Dimont manifest";
        minimum=1,
        maximum=DIMONT_MAX_LENGTH,
    )
    _validate_declared_model_shape(path, manifest, 5^(span + 1), motif_length, "Dimont")
    representation = _read_model_array(path, manifest, "representation")
    return Dimont(name, representation, span, motif_length)
end

function _read_slim_bundle(path::AbstractString, manifest::Dict, name::AbstractString)
    span = _required_manifest_int(
        manifest, "span", path, "Slim manifest"; minimum=0, maximum=SLIM_MAX_SPAN
    )
    motif_length = _required_manifest_int(
        manifest, "motif_length", path, "Slim manifest"; minimum=1, maximum=SLIM_MAX_LENGTH
    )
    _validate_declared_model_shape(path, manifest, 5^(span + 1), motif_length, "Slim")
    representation = _read_model_array(path, manifest, "representation")
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
