using Mimosa
using Printf

function median_elapsed(times::Vector{Float64})
    ordered = sort(times)
    midpoint = length(ordered) ÷ 2
    return if isodd(length(ordered))
        ordered[midpoint + 1]
    else
        (ordered[midpoint] + ordered[midpoint + 1]) / 2
    end
end

function timed_workload(query, targets, sequences, execution)
    prepared = prepare_profile(query, sequences; execution=execution)
    return [
        compare(
            prepared,
            target,
            sequences;
            execution=execution,
            metric=:co,
            search_range=10,
            window_radius=5,
            realign_window=3,
            min_logfpr=0.0f0,
        ) for target in targets
    ]
end

function main(args)
    length(args) == 4 || error("usage: cross_language_profile.jl FASTA MEME THREADS REPS")
    fasta, meme = args[1], args[2]
    threads = parse(Int, args[3])
    reps = parse(Int, args[4])
    threads <= Threads.nthreads() ||
        error("requested $threads threads, runtime has $(Threads.nthreads())")
    execution = threads == 1 ? SerialExecution() : ThreadedExecution(threads)

    sequences, _ = readsequences(fasta)
    models = [readmodel(meme; index=i) for i in 0:50]
    query, targets = first(models), models[2:end]

    length(timed_workload(query, targets, sequences, execution)) == 50 ||
        error("warm-up failed")
    times = Vector{Float64}(undef, reps)
    for i in eachindex(times)
        GC.gc()
        times[i] = @elapsed timed_workload(query, targets, sequences, execution)
    end
    @printf(
        "RESULT language=julia threads=%d median_s=%.9f min_s=%.9f julia_version=%s\n",
        threads,
        median_elapsed(times),
        minimum(times),
        VERSION,
    )
end

main(ARGS)
