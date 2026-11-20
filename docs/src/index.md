```@meta
CurrentModule = PracticalBayesianGenomics
```

# PracticalBayesianGenomics.jl

PracticalBayesianGenomics.jl builds Bayesian genomic models on [PracticalBayes.jl](https://github.com/EvoArt/PracticalBayes.jl). Use `bayes_alphabet_model` for BayesB or BayesC marker regression, or `animal_model` for breeding values with pedigree or genomic relatedness. Each builder returns a model and a blocked Gibbs sampler compatible with AbstractMCMC.

Install the package and AbstractMCMC in the same Julia environment:

```julia
using Pkg
Pkg.add(url="https://github.com/EvoArt/PracticalBayesianGenomics.jl")
Pkg.add("AbstractMCMC")
```

## Marker regression

Rows of `X` are individuals and columns are markers. Centre each marker column before fitting:

```julia
using PracticalBayesianGenomics, AbstractMCMC, Random, Statistics

rng = MersenneTwister(42)
X = randn(rng, 60, 12)
X .-= mean(X; dims=1)
y = 1.5 .* X[:, 3] .+ randn(rng, 60)

model, sampler = bayes_alphabet_model(y, X; variant=:C)
draw, state = AbstractMCMC.step(rng, model, sampler)
draw.δ  # one inclusion indicator per marker
draw.β  # marker effects
```

For an analysis, discard warm-up draws and summarize many later draws. The posterior mean of each `δ` is its inclusion probability. `variant=:B` selects the BayesB model.

## Animal model

`animal_model(y, X, Z, rel)` uses phenotype vector `y`, fixed-effect design `X`, incidence matrix `Z`, and a `RelatednessStructure`. For pedigree relatedness, parents must precede offspring and `0` denotes an unknown parent:

```julia
using SparseArrays, LinearAlgebra

sire = [0, 0, 1, 1, 3, 3]
dam  = [0, 0, 2, 2, 4, 4]
rel = pedigree_relatedness(sire, dam)
Z = sparse(1.0I, 6, 6)
Xfixed = ones(6, 1)
phenotype = [1.2, 0.9, 1.5, 1.7, 2.1, 1.8]
animal, animal_sampler = animal_model(phenotype, Xfixed, Z, rel)
```

Use `genomic_relatedness_from_markers` to build a relatedness structure from a dosage matrix instead. Inspect the model and its priors before interpreting posterior estimates.

## API reference

```@autodocs
Modules = [PracticalBayesianGenomics]
```
