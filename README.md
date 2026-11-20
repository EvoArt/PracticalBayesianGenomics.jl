# PracticalBayesianGenomics.jl

[![CI](https://github.com/EvoArt/PracticalBayesianGenomics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EvoArt/PracticalBayesianGenomics.jl/actions/workflows/CI.yml)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://EvoArt.github.io/PracticalBayesianGenomics.jl/dev/)

PracticalBayesianGenomics.jl supplies genomic analysis models for [PracticalBayes.jl](https://github.com/EvoArt/PracticalBayes.jl). It provides a Bayesian animal model using pedigree or genomic relatedness and BayesB/BayesC marker regression. The model builders return a model and a blocked Gibbs sampler for use with AbstractMCMC.

Install from the repository until registration is complete:

```julia
using Pkg
Pkg.add(url="https://github.com/EvoArt/PracticalBayesianGenomics.jl")
Pkg.add("AbstractMCMC")
```

Here is a small BayesC marker example. Rows are individuals and columns are markers.

```julia
using PracticalBayesianGenomics, AbstractMCMC, Random, Statistics

rng = MersenneTwister(42)
X = randn(rng, 60, 12)
X .-= mean(X; dims=1)             # centre each marker
y = 1.5 .* X[:, 3] .+ randn(rng, 60)

model, sampler = bayes_alphabet_model(y, X; variant=:C)
draw, state = AbstractMCMC.step(rng, model, sampler)
draw.δ                            # marker inclusion indicators
draw.β                            # marker effects
```

See the [documentation](https://EvoArt.github.io/PracticalBayesianGenomics.jl/dev/) for the animal model and API reference. The related [QuantitativeGenetics.jl](https://github.com/EvoArt/QuantitativeGenetics.jl) package fits frequentist REML animal models.
