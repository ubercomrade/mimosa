using Test
using Mimosa

@testset "Model storage round-trip: PWM" begin
    weights = Float32[
        0.5 -0.5 0.3 0.1
        -0.3 0.7 -0.2 0.2
        0.1 0.1 0.8 -0.3
        -0.2 0.3 -0.1 0.4
        -0.3 -0.3 -0.3 -0.3
    ]
    bg = (0.25f0, 0.25f0, 0.30f0, 0.20f0)
    pwm = PWM("test_pwm", weights, bg)

    dir = mktempdir()
    bundle_path = joinpath(dir, "pwm_bundle")

    writemodel(bundle_path, pwm)
    loaded = readmodel(bundle_path)

    @test loaded isa PWM
    @test loaded.name == pwm.name
    @test loaded.weights == pwm.weights
    @test loaded.background == pwm.background
end

@testset "Model storage round-trip: PFM" begin
    freq = Float32[
        0.3 0.2 0.4 0.1
        0.2 0.3 0.1 0.4
        0.1 0.4 0.2 0.3
        0.4 0.1 0.3 0.2
    ]
    pfm = PFM("test_pfm", freq)

    dir = mktempdir()
    bundle_path = joinpath(dir, "pfm_bundle")

    writemodel(bundle_path, pfm)
    loaded = readmodel(bundle_path)

    @test loaded isa PFM
    @test loaded.name == pfm.name
    @test loaded.frequencies == pfm.frequencies
end

@testset "Model storage round-trip: BaMM" begin
    # Order 1, 4 positions: 5^2 = 25 rows
    rep = Float32.(reshape(1.0:75, 25, 3))
    model = BaMM("test_bamm", rep, 1, 3)

    dir = mktempdir()
    bundle_path = joinpath(dir, "bamm_bundle")

    writemodel(bundle_path, model)
    loaded = readmodel(bundle_path)

    @test loaded isa BaMM
    @test loaded.name == model.name
    @test loaded.representation == model.representation
    @test loaded.order == model.order
    @test loaded.motif_length == model.motif_length
end

@testset "Model storage round-trip: SiteGA" begin
    rep = Float32.(reshape(1.0:75, 25, 3))
    model = SiteGA("test_sitega", rep, 3)

    dir = mktempdir()
    bundle_path = joinpath(dir, "sitega_bundle")

    writemodel(bundle_path, model)
    loaded = readmodel(bundle_path)

    @test loaded isa SiteGA
    @test loaded.name == model.name
    @test loaded.representation == model.representation
    @test loaded.motif_length == model.motif_length
end

@testset "Model storage round-trip: Dimont" begin
    # Span 1, 3 positions: 5^2 = 25 rows
    rep = Float32.(reshape(1.0:75, 25, 3))
    model = Dimont("test_dimont", rep, 1, 3)

    dir = mktempdir()
    bundle_path = joinpath(dir, "dimont_bundle")

    writemodel(bundle_path, model)
    loaded = readmodel(bundle_path)

    @test loaded isa Dimont
    @test loaded.name == model.name
    @test loaded.representation == model.representation
    @test loaded.span == model.span
    @test loaded.motif_length == model.motif_length
end

@testset "Model storage round-trip: Slim" begin
    rep = Float32.(reshape(1.0:75, 25, 3))
    model = Slim("test_slim", rep, 1, 3)

    dir = mktempdir()
    bundle_path = joinpath(dir, "slim_bundle")

    writemodel(bundle_path, model)
    loaded = readmodel(bundle_path)

    @test loaded isa Slim
    @test loaded.name == model.name
    @test loaded.representation == model.representation
    @test loaded.span == model.span
    @test loaded.motif_length == model.motif_length
end

@testset "Model storage: checksum validation" begin
    weights = Float32[
        0.5 -0.5
        -0.3 0.7
        0.1 0.1
        -0.2 0.3
        -0.3 -0.3
    ]
    bg = (0.25f0, 0.25f0, 0.25f0, 0.25f0)
    pwm = PWM("test", weights, bg)

    dir = mktempdir()
    bundle_path = joinpath(dir, "pwm_bundle")

    writemodel(bundle_path, pwm)

    # Corrupt the data file
    data_file = joinpath(bundle_path, "data", "weights.npy")
    write(data_file, UInt8[0, 0, 0, 0, 0])

    @test_throws MimosaError readmodel(bundle_path)
end

@testset "Model storage: unknown format" begin
    dir = mktempdir()
    bundle_path = joinpath(dir, "bad_bundle")
    mkpath(joinpath(bundle_path, "data"))

    # Write manifest with wrong format
    manifest_path = joinpath(bundle_path, "manifest.toml")
    write(
        manifest_path,
        """
format = "wrong"
format_version = 1
kind = "pwm"
name = "test"
""",
    )

    @test_throws MimosaError readmodel(bundle_path)
end

@testset "Model storage: version too high" begin
    weights = Float32[
        0.5 -0.5
        -0.3 0.7
        0.1 0.1
        -0.2 0.3
        -0.3 -0.3
    ]
    bg = (0.25f0, 0.25f0, 0.25f0, 0.25f0)
    pwm = PWM("test", weights, bg)

    dir = mktempdir()
    bundle_path = joinpath(dir, "pwm_bundle")

    writemodel(bundle_path, pwm)

    # Bump the version in manifest
    manifest_path = joinpath(bundle_path, "manifest.toml")
    content = read(manifest_path, String)
    content = replace(content, "format_version = 1" => "format_version = 999")
    write(manifest_path, content)

    @test_throws MimosaError readmodel(bundle_path)
end

@testset "readmodel: legacy fallback still works" begin
    # Ensure legacy file reading still works alongside bundle reading
    # by reading a .pfm file (non-directory path)
    dir = mktempdir()
    pfm_path = joinpath(dir, "test.pfm")
    write(
        pfm_path,
        """
0.3 0.2 0.4 0.1
0.2 0.3 0.1 0.4
0.1 0.4 0.2 0.3
0.4 0.1 0.3 0.2
""",
    )

    model = readmodel(pfm_path)
    @test model isa PWM
    @test size(model.weights) == (5, 4)
end
