using Test
using Mimosa
using JSON3

# Shared NPY reader for compatibility tests
include(joinpath(@__DIR__, "npy_reader.jl"))
using .NPYReader: read_npy

# Shared fixture metadata helper
const REPO_ROOT = joinpath(dirname(@__DIR__), "..")
const EXAMPLES = joinpath(REPO_ROOT, "examples")
const FIXTURE_COMPAT = joinpath(REPO_ROOT, "tests", "fixtures", "compatibility")
const _MANIFEST = JSON3.read(read(joinpath(FIXTURE_COMPAT, "manifest.json"), String))

function fixture_metadata(id::AbstractString)
    for f in _MANIFEST["fixtures"]
        f["id"] == id && return f["metadata"]
    end
    return error("fixture $id not found in manifest.")
end

# Aqua quality checks (optional, skip if Aqua not installed).
try
    using Aqua
    Aqua.test_all(
        Mimosa;
        ambiguities=false,
        unbound_args=false,
        stale_deps=false,
        project_extras=false,
    )
catch
    @info "Aqua not available, skipping quality checks."
end

@testset "Mimosa.jl Stage 1+2" begin
    # Unit tests
    include("unit/test_models.jl")
    include("unit/test_readers.jl")
    include("unit/test_metrics.jl")
    include("unit/test_alignment.jl")
    include("unit/test_serialization.jl")
    include("unit/test_sequences.jl")
    include("unit/test_profiles.jl")
    include("unit/test_sites.jl")

    # Property tests
    include("properties/test_properties.jl")

    # Compatibility tests against frozen oracle fixtures
    include("compatibility/test_oracle_fixtures.jl")
    include("compatibility/test_scan_fixtures.jl")
    include("compatibility/test_profile_fixtures.jl")
    include("compatibility/test_sites_fixtures.jl")

    # Integration tests (CLI path)
    include("integration/test_cli.jl")
end
