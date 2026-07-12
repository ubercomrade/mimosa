using Test
using Mimosa

const REPO_ROOT = joinpath(dirname(dirname(@__DIR__)), "..")
const EXAMPLES = joinpath(REPO_ROOT, "examples")

@testset "CLI main: motif comparison" begin
    query = joinpath(EXAMPLES, "pif4.meme")
    target = joinpath(EXAMPLES, "pif4.meme")
    code = Mimosa.main(["--query", query, "--target", target, "--metric", "pcc"])
    @test code == 0
end

@testset "CLI main: missing args returns 1" begin
    code = Mimosa.main(String[])
    @test code == 1
end

@testset "CLI main: nonexistent file returns 1" begin
    code = Mimosa.main(["--query", "/nope.meme", "--target", "/nope2.meme"])
    @test code == 1
end

@testset "Parallel null build: serial == threaded" begin
    # Build a small model collection and relations
    weights1 = Float32[
        0.5 -0.5 0.3
        -0.3 0.7 -0.2
        0.1 0.1 0.8
        -0.2 0.3 -0.1
        -0.3 -0.3 -0.3
    ]
    weights2 = Float32[
        0.3 0.2 0.5
        0.1 0.8 0.1
        0.2 0.3 0.4
        0.1 0.1 0.2
        -0.1 -0.1 -0.1
    ]
    weights3 = Float32[
        0.4 -0.4 0.2
        -0.2 0.6 -0.1
        0.2 0.2 0.7
        -0.1 0.4 0.0
        -0.2 -0.2 -0.2
    ]
    bg = (0.25f0, 0.25f0, 0.25f0, 0.25f0)
    m1 = PWM("m1", weights1, bg)
    m2 = PWM("m2", weights2, bg)
    m3 = PWM("m3", weights3, bg)

    models = [m1, m2, m3]

    # Relations: all in different groups → all pairs eligible
    tsv_content = "motif\tgroup\nm1\tA\nm2\tB\nm3\tC\n"
    dir = mktempdir()
    rel_path = joinpath(dir, "relations.tsv")
    write(rel_path, tsv_content)
    relations = parse_group_relations(rel_path; known_names=Set(["m1", "m2", "m3"]))

    # Serial null build
    serial_result = build_null(models, relations; execution=SerialExecution())
    serial_scores = serial_result.distribution.raw_scores

    # Threaded null build
    for nt in (1, 2, 4)
        threaded_result = build_null(models, relations; execution=ThreadedExecution(nt))
        threaded_scores = threaded_result.distribution.raw_scores
        @test threaded_scores == serial_scores
        @test threaded_result.total_comparisons == serial_result.total_comparisons
    end
end

@testset "Model storage: cross-format compatibility" begin
    # Write a model to bundle, read it back, and compare scan results
    pwm = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    dir = mktempdir()
    bundle = joinpath(dir, "pif4_bundle")
    writemodel(bundle, pwm)
    loaded = readmodel(bundle)

    seq = encode_sequence("ACGTACGTACGTACGTACGTACGTAC")
    original_scan = scan(pwm, seq; strands=ForwardOnly())
    loaded_scan = scan(loaded, seq; strands=ForwardOnly())
    @test original_scan == loaded_scan
end
