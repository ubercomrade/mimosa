# Quick Start

## Installation

The package is not yet registered in the General registry. Install from a local
clone:

```julia
using Pkg
Pkg.develop(path="/path/to/Mimosa.jl")
```

Or from the repository (once published):

```julia
using Pkg
Pkg.add(url="https://github.com/mimosa-jl/Mimosa.jl.git")
```

## Reading models

`readmodel` auto-detects format by file extension:

```jldoctest
julia> using Mimosa

julia> pwm = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> typeof(pwm)
PWM{Float32, Matrix{Float32}, NTuple{4, Float32}}

julia> pwm.name
"pwm_model"
```

For explicit control, use `read_meme`, `read_pfm`, `read_bamm`, `read_sitega`,
`read_dimont`, `read_slim`, or `read_scores`.

## Scanning sequences

`readsequences` returns a `(batch, names)` tuple:

```jldoctest
julia> using Mimosa

julia> batch, names = readsequences(joinpath(dirname(pwd()), "..", "examples", "foreground.fa"));

julia> typeof(batch)
EncodedSequenceBatch{Vector{UInt8}, Vector{Int64}}

julia> nsequences(batch)
100
```

Scan with different strand policies:

```jldoctest
julia> using Mimosa

julia> pwm = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> batch, _ = readsequences(joinpath(dirname(pwd()), "..", "examples", "foreground.fa"));

julia> scores = scan(pwm, batch; strands=ForwardOnly());

julia> typeof(scores)
RaggedArray{Float32, Vector{Float32}, Vector{Int64}}

julia> nrows(scores)
100
```

Threaded scan produces identical results to serial:

```jldoctest
julia> using Mimosa

julia> pwm = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> batch, _ = readsequences(joinpath(dirname(pwd()), "..", "examples", "foreground.fa"));

julia> scan(pwm, batch; execution=SerialExecution()) == scan(pwm, batch; execution=ThreadedExecution(4))
true
```

Generate random sequences for testing:

```jldoctest
julia> using Mimosa

julia> rb = make_random_sequences(10, 20; seed=42);

julia> typeof(rb)
EncodedSequenceBatch{Vector{UInt8}, Vector{Int64}}

julia> nsequences(rb)
10
```

## Comparing motifs

Direct matrix alignment with PCC, Euclidean distance, or cosine similarity:

```jldoctest
julia> using Mimosa

julia> pwm1 = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> pwm2 = readmodel(joinpath(dirname(pwd()), "..", "examples", "gata2.meme"));

julia> result = compare(pwm1, pwm2, sequences; metric=:co);

julia> typeof(result)
ComparisonResult

julia> result.metric
"pcc"
```

Serialize results to JSON:

```jldoctest
julia> using Mimosa

julia> pwm1 = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> pwm2 = readmodel(joinpath(dirname(pwd()), "..", "examples", "gata2.meme"));

julia> result = compare(pwm1, pwm2, sequences; metric=:co);

julia> startswith(to_json(result), "{")
true
```

## Profile comparison

Profile-based comparison scans sequences with each model, normalizes,
and compares the resulting score profiles:

```jldoctest
julia> using Mimosa

julia> pwm1 = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> pwm2 = readmodel(joinpath(dirname(pwd()), "..", "examples", "gata2.meme"));

julia> batch, _ = readsequences(joinpath(dirname(pwd()), "..", "examples", "foreground.fa"));

julia> result = compare(pwm1, pwm2, batch; metric=:co, search_range=10, window_radius=10);

julia> typeof(result)
ComparisonResult
```

One-to-many comparison with score profiles:

```jldoctest
julia> using Mimosa

julia> sp1 = ScoreProfile("q", build_ragged([Float32[0.1, 0.5, 0.3, 0.8]]));

julia> sp2 = ScoreProfile("t1", build_ragged([Float32[0.2, 0.4, 0.3, 0.7]]));

julia> sp3 = ScoreProfile("t2", build_ragged([Float32[0.3, 0.1, 0.9, 0.2]]));

julia> prepared = prepare_profile(sp1);

julia> results = compare(prepared, [sp2, sp3]; metric=:co, search_range=3, window_radius=2);

julia> length(results)
2
```

Available profile metrics: `:co`, `:co_rowwise`, `:dice`, `:dice_rowwise`, `:cosine`.

## Site extraction and PFM reconstruction

```jldoctest
julia> using Mimosa

julia> pwm = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> batch, _ = readsequences(joinpath(dirname(pwd()), "..", "examples", "foreground.fa"));

julia> sites = selectsites(pwm, batch, BestPerSequence(); strands=BestStrand());

julia> typeof(sites)
SiteCollection

julia> length(sites)
100
```

Reconstruct a PFM from selected sites:

```jldoctest
julia> using Mimosa

julia> pwm = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> batch, _ = readsequences(joinpath(dirname(pwd()), "..", "examples", "foreground.fa"));

julia> pfm = reconstruct_pfm(pwm, batch, BestPerSequence(); pseudocount=Float32(1e-4));

julia> eltype(pfm)
Float32

julia> size(pfm)
(4, 12)
```

## Null distributions

Build a null distribution from a collection of models:

```jldoctest
julia> using Mimosa

julia> bg = (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25));

julia> w1 = Float32[0.5 -0.5 0.3; -0.3 0.7 -0.2; 0.1 0.1 0.8; -0.2 0.3 -0.1; -0.3 -0.3 -0.3];

julia> w2 = Float32[0.3 0.2 0.5; 0.1 0.8 0.1; 0.2 0.3 0.4; 0.1 0.1 0.2; -0.1 -0.1 -0.1];

julia> w3 = Float32[0.4 -0.4 0.2; -0.2 0.6 -0.1; 0.2 0.2 0.7; -0.1 0.4 0.0; -0.2 -0.2 -0.2];

julia> models = [PWM("m1", w1, bg), PWM("m2", w2, bg), PWM("m3", w3, bg)];

julia> mkpath("/tmp/mimosa_quickstart");

julia> write("/tmp/mimosa_quickstart/rel.tsv", "motif\tgroup\nm1\tA\nm2\tB\nm3\tC\n");

julia> relations = parse_group_relations("/tmp/mimosa_quickstart/rel.tsv"; known_names=Set(["m1", "m2", "m3"]));

julia> null_result = build_null(models, relations; sequences=sequences, metric=:co);

julia> typeof(null_result.distribution)
NullDistribution

julia> null_result.distribution.strategy
"motif"

julia> null_result.distribution.metric
"pcc"
```

Compute p-values from the fitted GEV:

```jldoctest
julia> using Mimosa

julia> scores = [0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1, 0.9];

julia> fit = fit_gev(scores);

julia> typeof(fit)
GEVFit

julia> typeof(pvalue(fit, 0.5))
Float64
```

Save and load null distributions (portable bundle format):

```jldoctest
julia> using Mimosa

julia> bg = (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25));

julia> w1 = Float32[0.5 -0.5 0.3; -0.3 0.7 -0.2; 0.1 0.1 0.8; -0.2 0.3 -0.1; -0.3 -0.3 -0.3];

julia> w2 = Float32[0.3 0.2 0.5; 0.1 0.8 0.1; 0.2 0.3 0.4; 0.1 0.1 0.2; -0.1 -0.1 -0.1];

julia> w3 = Float32[0.4 -0.4 0.2; -0.2 0.6 -0.1; 0.2 0.2 0.7; -0.1 0.4 0.0; -0.2 -0.2 -0.2];

julia> models = [PWM("m1", w1, bg), PWM("m2", w2, bg), PWM("m3", w3, bg)];

julia> write("/tmp/mimosa_quickstart/rel.tsv", "motif\tgroup\nm1\tA\nm2\tB\nm3\tC\n");

julia> relations = parse_group_relations("/tmp/mimosa_quickstart/rel.tsv"; known_names=Set(["m1", "m2", "m3"]));

julia> null_result = build_null(models, relations; sequences=sequences, metric=:co);

julia> savenull("/tmp/mimosa_quickstart/null_dist", null_result.distribution);

julia> loaded = loadnull("/tmp/mimosa_quickstart/null_dist");

julia> loaded.strategy
"motif"
```

## Result annotation

Annotate comparison results with p-values from a null distribution:

```jldoctest
julia> using Mimosa

julia> bg = (Float32(0.25), Float32(0.25), Float32(0.25), Float32(0.25));

julia> w1 = Float32[0.5 -0.5 0.3; -0.3 0.7 -0.2; 0.1 0.1 0.8; -0.2 0.3 -0.1; -0.3 -0.3 -0.3];

julia> w2 = Float32[0.3 0.2 0.5; 0.1 0.8 0.1; 0.2 0.3 0.4; 0.1 0.1 0.2; -0.1 -0.1 -0.1];

julia> w3 = Float32[0.4 -0.4 0.2; -0.2 0.6 -0.1; 0.2 0.2 0.7; -0.1 0.4 0.0; -0.2 -0.2 -0.2];

julia> models = [PWM("m1", w1, bg), PWM("m2", w2, bg), PWM("m3", w3, bg)];

julia> write("/tmp/mimosa_quickstart/rel.tsv", "motif\tgroup\nm1\tA\nm2\tB\nm3\tC\n");

julia> relations = parse_group_relations("/tmp/mimosa_quickstart/rel.tsv"; known_names=Set(["m1", "m2", "m3"]));

julia> null_result = build_null(models, relations; sequences=sequences, metric=:co);

julia> result = compare(models[1], models[2], sequences; metric=:co);

julia> annotated = annotate_results([result], null_result.distribution; effective_number_of_targets=2);

julia> typeof(annotated[1])
AnnotatedResult
```

## Writing models

Write to portable Mimosa bundle (TOML manifest + NPY blobs):

```jldoctest
julia> using Mimosa

julia> pwm = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> mkpath("/tmp/mimosa_quickstart");

julia> writemodel("/tmp/mimosa_quickstart/pwm_bundle", pwm);

julia> loaded = readmodel("/tmp/mimosa_quickstart/pwm_bundle");

julia> typeof(loaded)
PWM{Float32, Matrix{Float32}, NTuple{4, Float32}}

julia> loaded.name
"pwm_model"
```

## Parallelism

```jldoctest
julia> using Mimosa

julia> pwm = readmodel(joinpath(dirname(pwd()), "..", "examples", "pif4.meme"));

julia> batch, _ = readsequences(joinpath(dirname(pwd()), "..", "examples", "foreground.fa"));

julia> r1 = scan(pwm, batch; execution=SerialExecution());

julia> r2 = scan(pwm, batch; execution=ThreadedExecution(4));

julia> r1 == r2
true
```
