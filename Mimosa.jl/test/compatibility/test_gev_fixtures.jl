using Test
using Mimosa
using .NPYReader: read_npy

# GEV compatibility tests against frozen Python oracle fixtures.
#
# The oracle fixtures store SciPy genextreme parameters in (c, loc, scale)
# order where c = -k (sign flipped from textbook k). Julia uses k = -c.
#
# Tolerance classes (from docs/numerical_compatibility.md):
# - statistical_fit: params atol=0.01, rtol=0.05 (different optimizers)
# - tail_probability: SF values atol=1e-4, rtol=1e-3

const GEV_FIXTURES = [
    (
        id="gev_fit_gumbel_200",
        scipy_c=0.041947262978483224,
        scipy_loc=0.08007903794590493,
        scipy_scale=1.0418170785546519,
    ),
    (
        id="gev_fit_normal_2000",
        scipy_c=0.2324262232042325,
        scipy_loc=-0.3616235942952878,
        scipy_scale=1.0070399602005424,
    ),
    (
        id="gev_fit_exponential_500",
        scipy_c=-0.4738193438982986,
        scipy_loc=0.4528977662681939,
        scipy_scale=0.46004988452665696,
    ),
    (
        id="gev_fit_uniform_5000",
        scipy_c=0.422675601019221,
        scipy_loc=0.4159473678472674,
        scipy_scale=0.29698195395605126,
    ),
]

function _fixture_entry(id::AbstractString)
    for f in _MANIFEST["fixtures"]
        f["id"] == id && return f
    end
    return error("fixture $id not found in manifest.")
end

@testset "GEV compatibility fixtures" begin
    for fixture in GEV_FIXTURES
        @testset "$(fixture.id)" begin
            entry = _fixture_entry(fixture.id)
            meta = entry["metadata"]
            arrays = entry["arrays"]

            # Load scores
            scores_path = joinpath(FIXTURE_COMPAT, arrays["scores"]["file"])
            scores = read_npy(scores_path)

            # Load SF query points and expected SF values
            sf_points_path = joinpath(FIXTURE_COMPAT, arrays["sf_points"]["file"])
            sf_values = read_npy(sf_points_path)
            query_points = meta["sf_query_points"]

            # Fit GEV
            result = fit_gev(scores; max_iter=2000, tol=1e-10)

            @test result isa GEVFit
            if result isa GEVFit
                # Compare parameters: Julia k = -c (SciPy)
                expected_k = -fixture.scipy_c
                expected_loc = fixture.scipy_loc
                expected_scale = fixture.scipy_scale

                # Parameters: different optimizers may converge to slightly
                # different points. Use generous tolerance.
                @test result.shape ≈ expected_k atol = 0.01 rtol = 0.05
                @test result.location ≈ expected_loc atol = 0.01 rtol = 0.05
                @test result.scale ≈ expected_scale atol = 0.01 rtol = 0.05

                # Compare survival function values at query points
                # SF should agree more closely than parameters
                for (i, x) in enumerate(query_points)
                    julia_sf = survival(result, x)
                    scipy_sf = sf_values[i]
                    @test julia_sf ≈ scipy_sf atol = 1e-4 rtol = 1e-3
                    @test 0.0 ≤ julia_sf ≤ 1.0
                end
            end
        end
    end
end
