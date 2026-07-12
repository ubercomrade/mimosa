# Quick Start

## Installation

```julia
using Pkg
Pkg.develop(path="/path/to/Mimosa.jl")
```

Or from the registry (once registered):

```julia
using Pkg
Pkg.add("Mimosa")
```

## Reading models

```julia
using Mimosa

# Read a PWM from MEME format
pwm = readmodel("examples/pif4.meme")

# Read from PFM format
pfm = readmodel("examples/gata2.pfm")

# Read BaMM, SiteGA, Dimont, Slim — auto-detected by extension
bamm = readmodel("examples/bamm_order2.ihbcp")
sitega = readmodel("examples/sitega.mat")
dimont = readmodel("examples/dimont.xml")
```

## Scanning sequences

```julia
# Read a FASTA file
batch = readsequences("examples/sequences.fa")

# Scan with different strand policies
scores_fwd = scan(pwm, batch; strands=ForwardOnly())
scores_best = scan(pwm, batch; strands=BestStrand())
scores_both = scan(pwm, batch; strands=BothStrands())

# Threaded batch scan
scores = scan(pwm, batch; strands=BestStrand(), execution=ThreadedExecution(4))
```

## Comparing motifs

```julia
# Direct matrix alignment
result = compare(pwm1, pwm2; metric=:pcc)
# => ComparisonResult("pif4", "MA0036.2", 0.4336f0, -1, "+-", "pcc")

# Serialize to JSON
println(to_json(result))
```

## Site extraction and PFM reconstruction

```julia
# Extract best site per sequence
sites = selectsites(pwm, batch, BestPerSequence(); strands=BestStrand())

# Reconstruct PFM from selected sites
pfm_new = reconstruct_pfm(pwm, batch, BestPerSequence(); pseudocount=1e-4f0)
```

## Null distributions

```julia
# Build a null distribution from a collection of models
models = [pwm1, pwm2, pwm3]
relations = parse_group_relations("examples/groups.tsv")
dist = build_null(models, relations; strategy="motif", metric="pcc")

# Compute p-values
p = pvalue(dist, 0.85)
e = evalue(p, 100)
adj = adjusted_pvalues([0.01, 0.02, 0.03]; method=BenjaminiHochberg())

# Save and load null distributions
savenull("null_dist", dist)
loaded = loadnull("null_dist")
```

## Writing models

```julia
# Write to portable Mimosa bundle (TOML manifest + NPY blobs)
writemodel("output/pwm_bundle", pwm)
loaded = readmodel("output/pwm_bundle")
```