# Benchmark: direct PWM comparison vs Python warm path (planned).

# Run with:
#   julia --project=Mimosa.jl/benchmark -e 'include("Mimosa.jl/benchmark/benchmarks.jl")'

using Mimosa
using BenchmarkTools

const REPO_ROOT = dirname(dirname(@__DIR__))
const EXAMPLES = joinpath(REPO_ROOT, "examples")

function setup()
    pwm1 = readmodel(joinpath(EXAMPLES, "pif4.meme"))
    pwm2 = readmodel(joinpath(EXAMPLES, "gata2.meme"))
    return (pwm1, pwm2)
end

function run_benchmarks()
    pwm1, pwm2 = setup()
    println("compare(pif4, gata2; pcc):")
    @btime compare($pwm1, $pwm2; metric="pcc")
    println("compare(pif4, pif4; pcc):")
    @btime compare($pwm1, $pwm1; metric="pcc")
end

run_benchmarks()