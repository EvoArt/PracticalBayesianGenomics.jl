# PracticalBayesianGenomics.jl

[![CI](https://github.com/EvoArt/PracticalBayesianGenomics.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/EvoArt/PracticalBayesianGenomics.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/EvoArt/PracticalBayesianGenomics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/EvoArt/PracticalBayesianGenomics.jl)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://EvoArt.github.io/PracticalBayesianGenomics.jl/dev/)

PracticalBayesianGenomics.jl supplies genomic analysis models for [PracticalBayes.jl](https://github.com/EvoArt/PracticalBayes.jl): a Bayesian animal model using pedigree, genomic or single-step relatedness, and BayesB and BayesCπ marker regression. Each model builder returns a model and a fully conjugate blocked Gibbs sampler for use with AbstractMCMC.

```julia
using Pkg
Pkg.add(["PracticalBayesianGenomics", "AbstractMCMC"])
```

Here is a small marker regression. Rows are individuals and columns are markers.

```julia
using PracticalBayesianGenomics, AbstractMCMC, Random, Statistics

rng = MersenneTwister(42)
X = randn(rng, 60, 12)
X .-= mean(X; dims=1)             # centre each marker
y = 1.5 .* X[:, 3] .+ randn(rng, 60)

model, sampler = bayes_alphabet_model(y, X; variant=:C)   # BayesCπ
draw, state = AbstractMCMC.step(rng, model, sampler)
draw.δ                            # marker inclusion indicators
draw.β                            # marker effects
```

BayesCπ (`variant=:C`) gives every included marker the same effect variance `σ²ᵦ`. BayesB (`variant=:B`) gives each marker its own effect variance, so draws carry the vector `τ²` instead of `σ²ᵦ`.

See the [documentation](https://EvoArt.github.io/PracticalBayesianGenomics.jl/dev/) for the animal model and API reference. The related [AnimalModels.jl](https://github.com/EvoArt/AnimalModels.jl) package fits the animal model by REML.
