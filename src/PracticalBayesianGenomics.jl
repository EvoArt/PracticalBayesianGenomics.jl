module PracticalBayesianGenomics

using PracticalBayes
using Distributions
using LinearAlgebra
using SparseArrays
using Random
using AbstractMCMC
using Statistics: var

include("variance_kernels.jl")
include("relatedness.jl")
include("bayes_alphabet.jl")
include("animal_model.jl")

# Variance-component and effect kernels
export InvGammaVarKernel, GaussianEffectKernel

# Relatedness structures
export RelatednessStructure,
    pedigree_relatedness, inbreeding_coefficients,
    genomic_relatedness, genomic_relatedness_from_markers,
    singlestep_relatedness

# Models
export bayes_alphabet_model, BayesAlphabetKernel,
    bayes_b_model, BayesBKernel,       # back-compat aliases
    animal_model, AnimalKernel

end # module PracticalBayesianGenomics
