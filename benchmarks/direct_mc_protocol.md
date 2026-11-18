# Direct Monte Carlo engineering protocol

This is an exploratory CPU reference, not a calibrated biological test or a
replacement for the existing pooled null. Fix the number of replicates, seed,
alphabet group, sequences, threshold and shifts before inspecting scores.
The ordered pair (A, B) uses the R-B null: independently relabel each motif
position of B, while A stays fixed. Full S4 and the subgroup preserving {A,T}
and {C,G} are distinct hypotheses. Swapping A and B changes the conditional
null, so symmetric p-values are not promised. Exchangeability under the chosen
orbit null is required; unrelated biology does not establish it.

One randomization per target identity and replicate is shared across query
rows. A SHA-256-derived RNG stream depends on the seed, model fingerprint and
replicate, never collection position. Duplicate identical models consequently
share randomizations. A target's random profile is prepared once per replicate
and compared to all query profiles; only counts and observed scores survive.
Each pair gets exactly K comparisons, including ties, and reports (k+1)/(K+1).
The minimum is a resolution, not a bound on the true tail probability.

Engineering acceptance: compare a pair alone and in different collections;
check explicit identity permutations and ties; compare native and generic
relabeling for PWM and BaMM; replay identical permutations on CPU and GPU,
including near ties. The same backend and numerical procedure must calculate
observed and null scores. Do not claim GPU equivalence or speed until scores
and exceedance counts agree, including adversarial ties. Report warmup and
steady timings, transfers and peak memory separately, with a profile-reusing
CPU baseline. Use small K for engineering checks; no tail fitting or long runs.

The present CPU reference keeps all query profiles and one target profile in
RAM. A GPU implementation should batch models and positions under an explicit
VRAM budget, reuse each prepared profile across its block of pairs, and retain
only pair counts, never K complete profile sets. Reloading query blocks may be
needed if the profiles exceed the budget. BaMM context mapping must follow
the native relabel implementation. Mixed types can use the same CPU path.
