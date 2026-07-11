using Test
using Mimosa

@testset "to_dict produces expected keys" begin
    result = ComparisonResult("q", "t", 1.0f0, 0, "++", "pcc")
    d = Mimosa.to_dict(result)
    @test Set(keys(d)) ==
        Set(["query", "target", "score", "offset", "orientation", "metric"])
    @test d["score"] == 1.0
    @test d["offset"] == 0
end

@testset "to_json round-trips query/target" begin
    result = ComparisonResult("pwm_model", "pwm_model", 1.0f0, 0, "++", "pcc")
    s = Mimosa.to_json(result)
    @test contains(s, "\"query\"")
    @test contains(s, "\"pwm_model\"")
    @test contains(s, "\"orientation\"")
    @test contains(s, "\"++\"")
    @test contains(s, "\"pcc\"")
end
