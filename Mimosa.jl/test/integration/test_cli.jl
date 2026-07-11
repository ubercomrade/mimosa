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
