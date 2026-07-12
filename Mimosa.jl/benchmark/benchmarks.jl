# Mimosa.jl Benchmark Suite — Stage 9
#
# Comprehensive benchmarks for PWM scanning, motif comparison, higher-order
# scanning, site extraction, GEV fitting, and threaded scaling.
#
# Run with:
#   julia --project=Mimosa.jl/benchmark -e 'include("Mimosa.jl/benchmark/benchmarks.jl")'
#
# Or from the repo root:
#   julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/benchmarks.jl
#
# Environment variables:
#   JULIA_NUM_THREADS  — controls thread count for threaded scaling tests

using Mimosa
using BenchmarkTools
using Printf
using Dates
using Random

const REPO_ROOT = dirname(dirname(@__DIR__))
const EXAMPLES = joinpath(REPO_ROOT, "examples")
const FIXTURES = joinpath(REPO_ROOT, "tests", "fixtures", "compatibility")

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

"""
    print_header(title)

Print a formatted section header.
"""
function print_header(title::AbstractString)
    println()
    println("=" ^ 60)
    println("  $title")
    return println("=" ^ 60)
end

"""
    bench(name, f, args...; kwargs...)

Run a benchmark and print a formatted result.
"""
function bench(name::AbstractString, f, args...; kwargs...)
    b = @benchmark $f($(args...); $(kwargs...))
    println(
        @sprintf(
            "  %-45s  %10.3f μs  ±%8.2f%%  allocs: %d",
            name,
            mean(b).time / 1000,
            std(b).time / mean(b).time * 100,
            b.allocs
        )
    )
    return b
end

# ──────────────────────────────────────────────────────────────────────────
# Setup: create models and sequences for benchmarking
# ──────────────────────────────────────────────────────────────────────────

function make_pwm(width::Int)
    rng = Random.MersenneTwister(42)
    weights = Matrix{Float32}(undef, 5, width)
    for col in 1:width
        # Generate random weights, ensure N row is the minimum
        for row in 1:4
            weights[row, col] = Float32(randn(rng) * 0.5)
        end
        weights[5, col] = minimum(@view weights[1:4, col]) - Float32(0.1)
    end
    bg = (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25))
    return PWM("bench_pwm_$width", weights, bg)
end

function make_bamm(order::Int, width::Int)
    n_rows = 5^(order + 1)
    rng = Random.MersenneTwister(43)
    rep = Matrix{Float32}(undef, n_rows, width)
    for i in 1:n_rows, j in 1:width
        rep[i, j] = Float32(randn(rng) * 0.3)
    end
    return BaMM("bench_bamm_o$order", rep, order, width)
end

# ──────────────────────────────────────────────────────────────────────────
# Benchmark groups
# ──────────────────────────────────────────────────────────────────────────

"""
    bench_pwm_scan()

Benchmark PWM scanning at different sizes and strand policies.
"""
function bench_pwm_scan()
    print_header("PWM Scanning (single sequence)")

    for width in (8, 15, 30)
        pwm = make_pwm(width)
        for seqlen in (100, 200, 1000)
            seq = make_random_sequences(1, seqlen; seed=42)
            enc = sequence(seq, 1)
            n_pos = npositions(seqlen, width)

            # Forward scan
            dest = Vector{Float32}(undef, n_pos)
            scan_forward!(dest, pwm.weights, enc, n_pos)  # warm up
            b = @benchmark scan_forward!($dest, $(pwm.weights), $enc, $n_pos)
            println(
                @sprintf(
                    "  scan_forward!  w=%d len=%d  %10.3f μs  allocs: %d",
                    width,
                    seqlen,
                    mean(b).time / 1000,
                    b.allocs
                )
            )

            # Best strand scan
            scan_best!(dest, pwm.weights, enc, n_pos)  # warm up
            b = @benchmark scan_best!($dest, $(pwm.weights), $enc, $n_pos)
            println(
                @sprintf(
                    "  scan_best!     w=%d len=%d  %10.3f μs  allocs: %d",
                    width,
                    seqlen,
                    mean(b).time / 1000,
                    b.allocs
                )
            )

            # Reverse complement
            rc = similar(enc)
            reverse_complement!(rc, enc)
            b = @benchmark reverse_complement!($rc, $enc)
            println(
                @sprintf(
                    "  rev_comp!      w=%d len=%d  %10.3f μs  allocs: %d",
                    width,
                    seqlen,
                    mean(b).time / 1000,
                    b.allocs
                )
            )
        end
        println()
    end
end

"""
    bench_pwm_batch_scan()

Benchmark batch PWM scanning (serial vs threaded).
"""
function bench_pwm_batch_scan()
    print_header("PWM Batch Scanning (serial vs threaded)")

    pwm = make_pwm(15)
    for (n_seqs, seq_len) in ((100, 200), (1000, 200), (10000, 200))
        batch = make_random_sequences(n_seqs, seq_len; seed=42)

        # Serial
        scan(pwm, batch; strands=BestStrand(), execution=SerialExecution())
        b_ser = @benchmark scan(
            $pwm, $batch; strands=BestStrand(), execution=SerialExecution()
        )
        println(
            @sprintf(
                "  serial    n=%d len=%d  %10.3f ms  allocs: %d",
                n_seqs,
                seq_len,
                mean(b_ser).time / 1e6,
                b_ser.allocs
            )
        )

        # Threaded (only if threads available)
        nthreads = max(1, Threads.nthreads())
        if nthreads > 1
            scan(pwm, batch; strands=BestStrand(), execution=ThreadedExecution(nthreads))
            b_thr = @benchmark scan(
                $pwm, $batch; strands=BestStrand(), execution=ThreadedExecution($nthreads)
            )
            println(
                @sprintf(
                    "  threaded  n=%d len=%d  %10.3f ms  allocs: %d  (x%.2f speedup, %d threads)",
                    n_seqs,
                    seq_len,
                    mean(b_thr).time / 1e6,
                    b_thr.allocs,
                    mean(b_ser).time / mean(b_thr).time,
                    nthreads
                )
            )
        end
        println()
    end
end

"""
    bench_motif_comparison()

Benchmark direct motif matrix comparison (all metrics, all orientations).
"""
function bench_motif_comparison()
    print_header("Motif Comparison (direct matrix alignment)")

    pwm1 = make_pwm(8)
    pwm2 = make_pwm(15)
    pwm3 = make_pwm(30)

    for (p1, p2, label) in (
        (pwm1, pwm1, "8×8"),
        (pwm1, pwm2, "8×15"),
        (pwm2, pwm3, "15×30"),
        (pwm1, pwm3, "8×30"),
    )
        for metric in (PearsonCorrelation(), EuclideanDistance(), CosineSimilarity())
            compare(p1, p2; metric=metric)  # warm up
            b = @benchmark compare($p1, $p2; metric=($metric))
            println(
                @sprintf(
                    "  %-8s  %-10s  %10.3f μs  allocs: %d",
                    label,
                    metric_name(metric),
                    mean(b).time / 1000,
                    b.allocs
                )
            )
        end
    end
    return println()
end

"""
    bench_higher_order_scan()

Benchmark higher-order model scanning (BaMM).
"""
function bench_higher_order_scan()
    print_header("Higher-Order Scanning (BaMM)")

    for order in (0, 1, 2, 3)
        width = 10
        model = make_bamm(order, width)
        seq = make_random_sequences(1, 200; seed=42)
        enc = sequence(seq, 1)
        kmer = order + 1
        ctx = order
        win = width + order
        n_pos = max(length(enc) - win + 1, 0)
        dest = Vector{Float32}(undef, n_pos)

        Mimosa._ho_scan_forward!(dest, model.representation, kmer, ctx, width, enc, n_pos)  # warm up
        b = @benchmark Mimosa._ho_scan_forward!(
            $dest, $(model.representation), $kmer, $ctx, $width, $enc, $n_pos
        )
        println(
            @sprintf(
                "  BaMM order=%d  kmer=%d  rows=%5d  %10.3f μs  allocs: %d",
                order,
                kmer,
                size(model.representation, 1),
                mean(b).time / 1000,
                b.allocs
            )
        )
    end
    return println()
end

"""
    bench_site_extraction()

Benchmark site extraction and PFM reconstruction.
"""
function bench_site_extraction()
    print_header("Site Extraction and PFM Reconstruction")

    pwm = make_pwm(15)
    for (n_seqs, seq_len) in ((100, 200), (1000, 200))
        batch = make_random_sequences(n_seqs, seq_len; seed=42)

        # Best-per-sequence selection
        selectsites(pwm, batch, BestPerSequence(); strands=BestStrand())
        b = @benchmark selectsites($pwm, $batch, BestPerSequence(); strands=BestStrand())
        println(
            @sprintf(
                "  selectsites(BestPerSequence)  n=%d  %10.3f μs  allocs: %d",
                n_seqs,
                mean(b).time / 1000,
                b.allocs
            )
        )

        # PFM reconstruction
        reconstruct_pfm(pwm, batch, BestPerSequence(); pseudocount=Float32(1e-4))
        b = @benchmark reconstruct_pfm(
            $pwm, $batch, BestPerSequence(); pseudocount=Float32(1e-4)
        )
        println(
            @sprintf(
                "  reconstruct_pfm              n=%d  %10.3f μs  allocs: %d",
                n_seqs,
                mean(b).time / 1000,
                b.allocs
            )
        )
    end
    return println()
end

"""
    bench_gev_fit()

Benchmark GEV fitting and p-value computation.
"""
function bench_gev_fit()
    print_header("GEV Fitting and Statistics")

    for n in (100, 500, 2000)
        rng = Random.MersenneTwister(42)
        samples = Float32.(randn(rng, n) .* 0.3 .+ 0.5)

        fit_gev(samples)  # warm up
        b = @benchmark fit_gev($samples)
        println(
            @sprintf(
                "  fit_gev(n=%d)  %10.3f μs  allocs: %d", n, mean(b).time / 1000, b.allocs
            )
        )
    end

    # BH FDR
    pvals = Float32.(rand(MersenneTwister(42), 1000))
    adjusted_pvalues(pvals; method=BenjaminiHochberg())
    b = @benchmark adjusted_pvalues($pvals; method=BenjaminiHochberg())
    println(
        @sprintf("  BH FDR(n=1000)  %10.3f μs  allocs: %d", mean(b).time / 1000, b.allocs)
    )

    return println()
end

"""
    bench_serial_vs_threaded_equivalence()

Verify that serial and threaded scanning produce identical results.
"""
function bench_serial_vs_threaded_equivalence()
    print_header("Serial vs Threaded Equivalence")

    pwm = make_pwm(15)
    batch = make_random_sequences(500, 200; seed=42)

    ser = scan(pwm, batch; strands=BestStrand(), execution=SerialExecution())
    nthreads = max(1, Threads.nthreads())

    for ntasks in (1, 2, 4)
        ntasks = min(ntasks, nthreads)
        thr = scan(pwm, batch; strands=BestStrand(), execution=ThreadedExecution(ntasks))
        identical = ser == thr
        println(
            @sprintf(
                "  serial == threaded(%d tasks): %s", ntasks, identical ? "✓" : "✗ MISMATCH"
            )
        )
    end
    return println()
end

"""
    bench_cli_latency()

Benchmark CLI argument parsing and JSON serialization latency.
"""
function bench_cli_latency()
    print_header("CLI and Serialization Latency")

    pwm = make_pwm(15)
    result = compare(pwm, pwm; metric=:pcc)

    to_json(result)
    b = @benchmark to_json($result)
    println(
        @sprintf(
            "  to_json(ComparisonResult)  %10.3f μs  allocs: %d",
            mean(b).time / 1000,
            b.allocs
        )
    )

    to_dict(result)
    b = @benchmark to_dict($result)
    println(
        @sprintf(
            "  to_dict(ComparisonResult)  %10.3f μs  allocs: %d",
            mean(b).time / 1000,
            b.allocs
        )
    )
    return println()
end

# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

"""
    main()

Run the full benchmark suite.

Prints results to stdout. To save results to a file, redirect stdout:
    julia --project=Mimosa.jl/benchmark Mimosa.jl/benchmark/benchmarks.jl > benchmark_results.txt
"""
function main()
    println("Mimosa.jl Benchmark Suite")
    println("Julia: ", VERSION)
    println("Threads: ", Threads.nthreads())
    println("Date: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))

    bench_pwm_scan()
    bench_pwm_batch_scan()
    bench_motif_comparison()
    bench_higher_order_scan()
    bench_site_extraction()
    bench_gev_fit()
    bench_serial_vs_threaded_equivalence()
    bench_cli_latency()

    println("=" ^ 60)
    println("  Benchmark suite complete.")
    return println("=" ^ 60)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
