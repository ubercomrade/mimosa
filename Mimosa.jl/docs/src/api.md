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
read_fasta
```

## Scanning

```@docs
scan
scan!
scorebounds
npositions
StrandPair
```

## Sequence representation

```@docs
EncodedSequenceBatch
encode_sequence
reverse_complement
reverse_complement!
make_random_sequences
RaggedArray
```

## Strand policies

```@docs
StrandPolicy
ForwardOnly
ReverseOnly
BestStrand
BothStrands
```

## Profile comparison

```@docs
compare
ComparisonResult
AbstractProfileMetric
OverlapCoefficient
OverlapCoefficientRowwise
DiceSimilarity
DiceSimilarityRowwise
CosineSimilarityProfile
ProfileConfig
PreparedProfile
prepare_profile
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
GEVFitResult
fit_gev
survival
cdf
pvalue
BenjaminiHochberg
adjusted_pvalues
evalue
NullDistribution
NullBuildConfig
NullBuildResult
build_null
annotate_results
AnnotatedResult
parse_group_relations
GroupRelations
savenull
loadnull
```

## Profile comparison

```@docs
AbstractProfileMetric
OverlapCoefficient
OverlapCoefficientRowwise
DiceSimilarity
DiceSimilarityRowwise
CosineSimilarityProfile
PreparedProfile
prepare_profile
ProfileConfig
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
