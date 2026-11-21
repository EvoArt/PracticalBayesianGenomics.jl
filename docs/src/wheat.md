# Example: which markers matter for wheat yield?

599 CIMMYT wheat lines genotyped at 1,279 DArT markers, with grain yield (Crossa et
al. 2010, distributed with the R package BGLR). The lines are inbred, so a marker's
presence/absence is a 0/2 dosage.

```@example wheat
using PracticalBayesianGenomics, RData, Statistics
using Random: Xoshiro
import AbstractMCMC, CodecXz, Downloads      # BGLR's .RData files are xz-compressed

file = joinpath(tempdir(), "wheat.RData")
isfile(file) || Downloads.download("https://github.com/gdlc/BGLR-R/raw/master/data/wheat.RData", file)
wheat = load(file)

markers = 2 .* Float64.(wheat["wheat.X"])
yield = Float64.(wheat["wheat.Y"][:, 1])     # environment 1, standardised
size(markers)
```

## BayesC

BayesC assumes most markers have no effect and estimates, for each marker, the
posterior probability that it is in the model. Centre the marker columns and the
phenotype, then sample; `discard_initial` drops the warm-up draws.

```@example wheat
Xc = markers .- mean(markers; dims=1)
model, sampler = bayes_alphabet_model(yield .- mean(yield), Xc; variant=:C)
draws = AbstractMCMC.sample(Xoshiro(1), model, sampler, 2000; discard_initial=1000, progress=false)

inclusion = mean(draws[:δ])                 # posterior inclusion probability per marker
top = sortperm(inclusion; rev=true)[1:8]
[top inclusion[top]]
```

Yield is highly polygenic: no marker stands out on its own, and the signal is spread
thinly across the genome. That is why genomic BLUP, which assumes exactly this, predicts
it well; see the
[QuantitativeGenetics example](https://evoart.github.io/QuantitativeGenetics.jl/dev/wheat/)
for REML and held-out prediction on the same lines. `variant=:B` fits BayesB.
