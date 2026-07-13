# Execution policies for top-level parallelism.
#
# Parallelism in Mimosa is applied at the *top independent level* — sequences,
# target models, or comparison pairs — never inside inner scanning kernels.
# The kernels themselves are serial and composable. This keeps results
# deterministic regardless of thread count.
#
# Design per ADR 0004 (parallelism-and-rng):
#   - `SerialExecution` processes items in order.
#   - `ThreadedExecution(ntasks)` partitions items into `ntasks` chunks,
#     each processed by a separate Julia task. Results are written into
#     pre-allocated slots indexed by original position, so the output order
#     is independent of scheduling.
#   - Nested parallelism is controlled via `._parallel_depth` to prevent
#     uncontrolled thread spawning.

"""
    ExecutionPolicy

Abstract supertype for parallel execution policies.

Concrete policies:
- [`SerialExecution`](@ref): process items sequentially (default).
- [`ThreadedExecution`](@ref): process items using multiple Julia threads.
"""
abstract type ExecutionPolicy end

"""
    SerialExecution

Execute items in sequential order. This is the default policy and guarantees
deterministic, single-threaded execution.
"""
struct SerialExecution <: ExecutionPolicy end

"""
    ThreadedExecution

Execute items using multiple Julia tasks running on separate threads.

Fields:
- `ntasks::Int`: maximum number of concurrent tasks. Defaults to
  `Threads.nthreads()` when constructed with no arguments.

Results are written into pre-allocated slots indexed by original position,
so the output order and values are identical to `SerialExecution`.
"""
struct ThreadedExecution <: ExecutionPolicy
    ntasks::Int

    function ThreadedExecution(ntasks::Integer)
        ntasks < 1 && throw(ArgumentError("ntasks must be ≥ 1, got $ntasks."))
        return new(Int(ntasks))
    end
end

ThreadedExecution() = ThreadedExecution(max(1, Threads.nthreads()))

function Base.show(io::IO, ::SerialExecution)
    return print(io, "SerialExecution()")
end

function Base.show(io::IO, pol::ThreadedExecution)
    return print(io, "ThreadedExecution(ntasks=$(pol.ntasks))")
end

# ── Parallel map helper ───────────────────────────────────────────────────
#
# `_parallel_for(policy, n, f!)` — iterate `f!(i)` for `i in 1:n`.
# Under `SerialExecution` this is a simple loop. Under `ThreadedExecution`
# it partitions the range `1:n` into `ntasks` contiguous chunks and spawns
# a task per chunk. Each task processes its chunk sequentially.
#
# The caller is responsible for pre-allocating result slots and ensuring
# `f!` is thread-safe (no shared mutable state, no `push!` to shared vectors).

"""
    _parallel_for(f!, policy::ExecutionPolicy, n::Int)

Execute `f!(i)` for each `i in 1:n` according to `policy`.

Under `SerialExecution`, this is a simple `for` loop. Under
`ThreadedExecution`, the range is split into contiguous chunks processed by
separate tasks. The function `f!` must be thread-safe: no shared mutable
state, no `push!` to shared vectors, results written only to pre-allocated
slots indexed by `i`.
"""
function _parallel_for end

function _parallel_for(f!, ::SerialExecution, n::Int)
    @inbounds for i in 1:n
        f!(i)
    end
    return nothing
end

function _parallel_for(f!, pol::ThreadedExecution, n::Int)
    ntasks = min(pol.ntasks, n)
    ntasks <= 1 && return _parallel_for(f!, SerialExecution(), n)

    # A bounded queue avoids stranding a worker behind a long ragged item.
    next_index = Threads.Atomic{Int}(1)
    @sync for t in 1:ntasks
        Threads.@spawn begin
            while true
                i = Threads.atomic_add!(next_index, 1)
                i > n && break
                f!(i)
            end
        end
    end

    return nothing
end
