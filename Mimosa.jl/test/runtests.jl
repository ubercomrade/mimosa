using Test
using Mimosa
using JSON3

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

@testset "Mimosa.jl Stage 1" begin
    # Unit tests
    include("unit/test_models.jl")
    include("unit/test_readers.jl")
    include("unit/test_metrics.jl")
    include("unit/test_alignment.jl")
    include("unit/test_serialization.jl")

    # Property tests
    include("properties/test_properties.jl")

    # Compatibility tests against frozen oracle fixtures
    include("compatibility/test_oracle_fixtures.jl")

    # Integration tests (CLI path)
    include("integration/test_cli.jl")
end
