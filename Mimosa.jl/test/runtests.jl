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
const _MANIFEST_PATH = joinpath(FIXTURE_COMPAT, "manifest.json")
const _HAS_COMPAT_FIXTURES = isfile(_MANIFEST_PATH)
const _MANIFEST = _HAS_COMPAT_FIXTURES ? JSON3.read(read(_MANIFEST_PATH, String)) : nothing

function fixture_metadata(id::AbstractString)
    _HAS_COMPAT_FIXTURES || error("compatibility fixtures are unavailable: $FIXTURE_COMPAT")
    for f in _MANIFEST["fixtures"]
        f["id"] == id && return f["metadata"]
    end
    return error("fixture $id not found in manifest.")
end

# Aqua quality checks — fail-closed (Aqua is a required test dependency).
using Aqua
Aqua.test_all(Mimosa; stale_deps=false, project_extras=false)

@testset "Mimosa.jl Stage 1-7" begin
    # Unit tests
    include("unit/test_models.jl")
    include("unit/test_readers.jl")
    include("unit/test_serialization.jl")
    include("unit/test_sequences.jl")
    include("unit/test_profiles.jl")
    include("unit/test_sites.jl")
    include("unit/test_bamm.jl")
    include("unit/test_sitega.jl")
    include("unit/test_dimont.jl")
    include("unit/test_slim.jl")
    include("unit/test_gev.jl")
    include("unit/test_pvalues.jl")
    include("unit/test_relations.jl")
    include("unit/test_null_distribution.jl")
    include("unit/test_null_storage.jl")
    include("unit/test_parallel.jl")
    include("unit/test_cache.jl")
    include("unit/test_model_storage.jl")
    include("unit/test_validation.jl")
    include("unit/test_exports.jl")
    include("unit/test_type_stability.jl")

    # Property tests
    include("properties/test_properties.jl")

    # Compatibility fixtures are an explicitly separate contract. The root
    # Python corpus may be absent in a source checkout, but must not prevent
    # independent unit/security tests from running.
    if _HAS_COMPAT_FIXTURES
        include("compatibility/test_oracle_fixtures.jl")
        include("compatibility/test_scan_fixtures.jl")
        include("compatibility/test_profile_fixtures.jl")
        include("compatibility/test_sites_fixtures.jl")
        include("compatibility/test_bamm_fixtures.jl")
        include("compatibility/test_sitega_fixtures.jl")
        include("compatibility/test_dimont_fixtures.jl")
        include("compatibility/test_slim_fixtures.jl")
        include("compatibility/test_gev_fixtures.jl")
    else
        @test_skip "compatibility fixture corpus is unavailable; compatibility contract not checked"
    end

    # Integration tests (CLI path)
    include("integration/test_cli.jl")
    include("integration/test_cli_subprocess.jl")

    # JET static analysis (fail-closed for type instability in hot paths)
    include("jet/test_jet.jl")
end

# Downstream contract test — runs in a separate consumer environment:
#   julia --project=test/downstream test/downstream/runtests.jl
