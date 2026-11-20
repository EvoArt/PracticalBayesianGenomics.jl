using PracticalBayes
using Distributions
using LinearAlgebra

# ---------------------------------------------------------------------------
# Conjugate variance-component kernels.
#
# Every variance component in a Gaussian animal / Bayesian-alphabet model has a
# closed-form full conditional in the scaled-inverse-χ² (equivalently
# Inverse-Gamma) family: given the rest of the state, resampling a variance is
# just "prior degrees-of-freedom + a sum of squares read off the current draw."
# Routing these through NUTS (as the first stubs did) is both slower and worse-
# mixing than drawing the exact conjugate value, and it forces an AD gradient
# and an HMC sub-chain for a quantity that has a one-line closed form. These
# kernels are the shared efficiency win across BOTH the animal model and the
# Bayesian alphabet.
#
# They are the Inverse-Gamma sibling of PracticalBayes' own `ConjugateGibbs`
# (which ships :beta / :dirichlet). Same contract: an `AbstractGibbsKernel`
# that `Gibbs` calls once per sweep, outside every gradient step, so plain
# Float64 arithmetic is fine throughout. The model-specific part — WHAT sum of
# squares to accumulate — is supplied by the user as a closure, exactly as
# `ConjugateGibbs`'s `count` closure is.
# ---------------------------------------------------------------------------

"""
    InvGammaVarKernel(name, prior, ss; predictor=nothing)

An `AbstractGibbsKernel` (see `PracticalBayes.gibbs_step`) that resamples the
variance-component block `name` from its conjugate Inverse-Gamma full
conditional.

The model must place a `name ~ InverseGamma(ν/2, ν·S²/2)` prior on the variance
(the scaled-inverse-χ²(ν, S²) parameterisation used throughout this package).
`prior = (ν, S²)` names those same prior hyperparameters; the kernel adds the
data's degrees of freedom and sum of squares and draws

    name ~ InverseGamma((ν + n)/2, (ν·S² + SS)/2)

where `n` and `SS` come from the user's `ss` closure.

Arguments:

- `name::Symbol` — the variance block this kernel owns and returns.
- `prior` — `(ν, S²)`: the prior degrees of freedom and scale, matching the
  `InverseGamma(ν/2, ν·S²/2)` written in the model body.
- `ss` — the sufficient-statistics closure `ss(c) -> (n, SS)`, given the
  `ModelConditional` `c`. `n` is the number of terms (residual d.f. or number of
  random effects) and `SS` their sum of squares. The closure does ALL model-
  specific work — it reaches into `c.values` / `c.model.args` itself — so this
  kernel imposes no assumptions about the model's data layout.

# Example — residual variance of a linear model with mean `μ = X·β`:

```julia
k = InvGammaVarKernel(:σ²ₑ, (νe, Se2)) do c
    y = c.model.args.y
    X = c.model.args.X
    resid = y .- X * c.values.β
    (length(y), dot(resid, resid))
end
# ... spl = Gibbs(:σ²ₑ => k, :β => NUTS(0.8), ...)
```
"""
struct InvGammaVarKernel{F} <: PracticalBayes.AbstractGibbsKernel
    name::Symbol
    prior::Tuple{Float64,Float64}   # (ν, S²)
    ss::F
end
InvGammaVarKernel(name::Symbol, prior; ss) = InvGammaVarKernel(name, (Float64(prior[1]), Float64(prior[2])), ss)
# ss-first method so the `do`-block form reads naturally (the closure is passed
# as the first positional argument by Julia's `do` syntax), matching
# `ConjugateGibbs`'s own count-first constructor.
InvGammaVarKernel(ss, name::Symbol, prior) = InvGammaVarKernel(name, (Float64(prior[1]), Float64(prior[2])), ss)

function PracticalBayes.gibbs_step(rng, k::InvGammaVarKernel, block_names, c::PracticalBayes.ModelConditional)
    block_names == (k.name,) ||
        error("InvGammaVarKernel owns block $(k.name), got $(block_names)")
    ν, S2 = k.prior
    n, SS = k.ss(c)
    # Posterior scaled-inverse-χ²(ν + n, (ν·S² + SS)/(ν + n)), written in the
    # InverseGamma(shape, scale) parameterisation Distributions.jl expects.
    shape = (ν + n) / 2
    scale = (ν * S2 + SS) / 2
    draw = rand(rng, InverseGamma(shape, scale))
    return NamedTuple{(k.name,)}((draw,))
end
