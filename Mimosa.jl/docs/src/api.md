# Julia API Reference

## Model types

```@docs
AbstractMotifModel
PFM
PWM
BaMM
SiteGA
Dimont
Slim
ScoreProfile
```

## Model I/O

```@docs
readmodel
writemodel
readsequences
```

## Scanning

```@docs
scan
scan!
scorebounds
npositions
```

## Sequence representation

```@docs
EncodedSequenceBatch
encode_sequence
reverse_complement
reverse_complement!
make_random_sequences
```

## Strand policies

```@docs
StrandPolicy
ForwardOnly
ReverseOnly
BestStrand
BothStrands
```

## Motif comparison

```@docs
compare
ComparisonResult
PearsonCorrelation
EuclideanDistance
CosineSimilarity
```

## Site extraction

```@docs
SiteSelector
BestPerSequence
ThresholdHits
TopFractionHits
SiteHit
SiteCollection
selectsites
reconstruct_pfm
```

## Statistics

```@docs
GEVFit
GEVFitFailure
fit_gev
survival
cdf
pvalue
BenjaminiHochberg
adjusted_pvalues
evalue
NullDistribution
build_null
savenull
loadnull
```

## Execution policies

```@docs
ExecutionPolicy
SerialExecution
ThreadedExecution
```

## Cache

```@docs
Cache
cache_key
cache_has
cache_get
cache_set
clearcache
```

## Serialization

```@docs
to_json
to_dict
```

## Errors

```@docs
MimosaError
ModelFormatError
ModelDimensionError
InvariantError
```