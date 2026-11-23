using PracticalBayes
using Distributions
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# The Bayesian alphabet (BayesB, BayesCπ) for whole-genome regression
#
#     y = Xβ + e,   e ~ N(0, σ²ₑ I)
#
# with a spike-and-slab prior on the p marker effects β. The workhorse is a
# single-site Gibbs scan over markers, sampling each inclusion indicator δⱼ
# (with βⱼ integrated out) and then βⱼ from its exact conditional, while keeping
# a running residual that is rank-1 updated as each βⱼ changes.
#
# EFFICIENCY: the per-column cross-products xⱼ'xⱼ are CONSTANT across sweeps and
# are precomputed once into the kernel (`xtx`), and columns are used as views
# into X. Residual bookkeeping is incremental (`e .-= xⱼ·Δβⱼ`), so each marker
# update is O(n), giving an O(n·p) sweep with no O(n·p) matrix rebuild.
#
# Variants:
#   * BayesCπ: one effect variance σ²ᵦ shared by every included marker
#   * BayesB:  each marker j has its own effect variance τ²ⱼ (Meuwissen et al. 2001)
# Blocks:
#   * (β, δ) or (β, δ, τ²)          → BayesAlphabetKernel (this file)
#   * variance components σ²ᵦ, σ²ₑ → conjugate Inverse-Gamma (InvGammaVarKernel)
#   * exclusion probability π       → conjugate Beta (PracticalBayes.ConjugateGibbs)
# ---------------------------------------------------------------------------

"""
    BayesAlphabetKernel(xtx; variant=:C, νb=3.0, S2b=1.0)

Gibbs kernel for the marker-effect block of [`bayes_alphabet_model`](@ref).
`xtx` is the precomputed vector of column cross-products `xⱼ'xⱼ`. Each marker's
inclusion indicator `δⱼ` is drawn with `βⱼ` integrated out, then `βⱼ` is drawn
from its normal conditional (`βⱼ = 0` when `δⱼ = 0`).

- `:C` (default) — BayesCπ. Included markers share one effect variance `σ²ᵦ`,
  read from the current state. The kernel owns the block `(:β, :δ)`; `νb` and
  `S2b` are unused.
- `:B` — BayesB. Marker `j` has its own effect variance `τ²ⱼ` with a
  scaled-inverse-χ²(`νb`, `S2b`) prior. After the marker scan, each `τ²ⱼ` is
  drawn from its conjugate conditional: scaled-inverse-χ²(`νb + 1`,
  `(νb·S2b + βⱼ²)/(νb + 1)`) when `δⱼ = 1`, and the prior when `δⱼ = 0`. The
  kernel owns the block `(:β, :δ, :τ²)`.

Both variants read `y`, `X` from the model arguments and `σ²ₑ`, `π` from the
current state. Prefer building via [`bayes_alphabet_model`](@ref).
"""
mutable struct BayesAlphabetKernel{V<:AbstractVector} <: PracticalBayes.AbstractGibbsKernel
    const xtx::V
    const variant::Symbol
    const νb::Float64
    const S2b::Float64
    resid::Vector{Float64}   # residual y − Xβ at the END of the last sweep's scan
end
function BayesAlphabetKernel(xtx; variant::Symbol=:C, νb::Real=3.0, S2b::Real=1.0)
    variant in (:B, :C) || throw(ArgumentError("variant must be :B or :C, got :$variant"))
    return BayesAlphabetKernel(xtx, variant, Float64(νb), Float64(S2b), Float64[])
end

_slabvar(v::Real, j) = v
_slabvar(v::AbstractVector, j) = @inbounds v[j]

function PracticalBayes.gibbs_step(rng, k::BayesAlphabetKernel, block_names, c::PracticalBayes.ModelConditional)
    y = c.model.args.y
    X = c.model.args.X
    β = copy(c.values.β)
    δ = copy(c.values.δ)
    if k.variant === :B
        block_names == (:β, :δ, :τ²) ||
            error("BayesAlphabetKernel(variant=:B) owns the (:β, :δ, :τ²) block, got $(block_names)")
        τ² = copy(c.values.τ²)
        e = _marker_scan!(rng, k, β, δ, τ², y, X, c.values.σ²ₑ, c.values.π)
        # τ²ⱼ | βⱼ, δⱼ: conjugate update for included markers, prior otherwise.
        a0, b0 = k.νb / 2, k.νb * k.S2b / 2
        @inbounds for j in eachindex(τ²)
            τ²[j] = δ[j] == 1 ? rand(rng, InverseGamma(a0 + 0.5, b0 + β[j]^2 / 2)) :
                                rand(rng, InverseGamma(a0, b0))
        end
        k.resid = e
        return (; β = β, δ = δ, τ² = τ²)
    else
        block_names == (:β, :δ) ||
            error("BayesAlphabetKernel(variant=:C) owns the (:β, :δ) block, got $(block_names)")
        e = _marker_scan!(rng, k, β, δ, c.values.σ²ᵦ, y, X, c.values.σ²ₑ, c.values.π)
        k.resid = e
        return (; β = β, δ = δ)
    end
end

# Single-site scan over markers. `slab` is the shared σ²ᵦ (BayesCπ) or the
# vector of per-marker τ²ⱼ (BayesB). Updates β and δ in place; returns y − Xβ.
function _marker_scan!(rng, k::BayesAlphabetKernel, β, δ, slab, y, X, σ²ₑ, π)
    p = size(X, 2)

    # Track the residual e = y − Xβ incrementally, so each marker conditional
    # needs one O(n) inner product, `dot(xⱼ, e)`. The initial residual uses only
    # the included markers, so its cost is O(n·‖δ‖₀) rather than O(n·p).
    e = Vector{Float64}(y)
    @inbounds for j in 1:p
        bj = β[j]
        if δ[j] == 1 && bj != 0.0
            @. e -= @view(X[:, j]) * bj
        end
    end

    @inbounds for j in 1:p
        β_prev = δ[j] == 1 ? β[j] : 0.0
        xj = view(X, :, j)
        xj2 = k.xtx[j]
        v = _slabvar(slab, j)           # this marker's slab variance

        # xⱼ'r₋ⱼ where r₋ⱼ = e + xⱼ·βⱼ is the residual with marker j removed.
        xj_r = dot(xj, e) + xj2 * β_prev

        βhat = xj_r / xj2
        σ²0 = σ²ₑ / xj2                 # sampling variance of βhat
        σ²1 = v + σ²0                   # marginal variance of βhat when included

        # log posterior odds of inclusion (δ=1) vs exclusion (δ=0)
        p0 = log(π) + logpdf(Normal(0.0, sqrt(σ²0)), βhat)
        p1 = log1p(-π) + logpdf(Normal(0.0, sqrt(σ²1)), βhat)
        prob = clamp(1 / (1 + exp(p0 - p1)), 1e-12, 1.0 - 1e-12)

        if rand(rng) < prob
            δ[j] = 1
            μj = v * xj_r / (σ²ₑ + v * xj2)
            σ²j = v * σ²ₑ / (σ²ₑ + v * xj2)
            β[j] = μj + sqrt(σ²j) * randn(rng)
        else
            δ[j] = 0
            β[j] = 0.0
        end

        Δ = β[j] - β_prev
        if Δ != 0.0
            @. e -= xj * Δ             # rank-1 update of the residual
        end
    end
    return e
end

"""
    bayes_alphabet_model(y, X; variant=:C, b=1.0, c=1.0,
                         S2b=nothing, S2e=var(y), νb=3.0, νe=3.0)
        -> (model, sampler)

Build a Bayesian-alphabet whole-genome regression and its blocked-Gibbs sampler.

- `y`, `X` — `n` phenotypes and `n×p` (centred) marker matrix.
- `variant` — `:C` for BayesCπ (one effect variance `σ²ᵦ` shared by the
  included markers) or `:B` for BayesB (a separate effect variance `τ²ⱼ` for
  each marker). See [`BayesAlphabetKernel`](@ref).
- `(b, c)` — Beta prior for `π`, the probability that a marker is excluded.
- `(νb, S2b)` — scaled-inv-χ² prior for the effect variance: `σ²ᵦ` under
  BayesCπ, each `τ²ⱼ` under BayesB. If `S2b` is not given, BayesCπ uses
  `0.1·var(y)`. BayesB follows BGLR's default and sets the prior mode of `τ²ⱼ`
  to half of `var(y)` shared among the expected number of included markers:
  `S2b = 0.5·var(y) / (MSx·c/(b+c)) · (νb+2)/νb`, with `MSx = Σⱼ xⱼ'xⱼ / n`.
- `(νe, S2e)` — scaled-inv-χ² prior for `σ²ₑ`.

Returns `(model, spl)`. Draws contain `β`, `δ`, `σ²ₑ`, `π`, and either `σ²ᵦ`
(BayesCπ) or the vector `τ²` (BayesB). Sample with
`AbstractMCMC.sample(model, spl, N)`.
"""
function bayes_alphabet_model(y, X; variant::Symbol=:C,
                              b=1.0, c=1.0,
                              S2b=nothing, S2e=var(y), νb=3.0, νe=3.0)
    variant in (:B, :C) || throw(ArgumentError("variant must be :B or :C, got :$variant"))
    n, p = size(X)
    xtx = [dot(view(X, :, j), view(X, :, j)) for j in 1:p]
    if S2b === nothing
        S2b = variant === :C ? var(y) * 0.1 :
              0.5 * var(y) / (sum(xtx) / n * (c / (b + c))) * (νb + 2) / νb
    end

    βδ_kernel = BayesAlphabetKernel(xtx; variant=variant, νb=νb, S2b=S2b)
    σ²ₑ_kernel = InvGammaVarKernel(:σ²ₑ, (νe, S2e)) do cc
        # Reuse the residual y − Xβ the marker kernel computed on this sweep. The
        # Gibbs scan updates the marker block before σ²ₑ and nothing in between
        # changes β, so the stash is current. Falls back to a direct compute.
        r = βδ_kernel.resid
        if isempty(r)
            r = cc.model.args.y .- cc.model.args.X * (cc.values.β .* cc.values.δ)
        end
        (length(r), dot(r, r))
    end
    # Conjugate Beta for π: δⱼ ~ Bernoulli(1 − π), so excluded markers count
    # towards b and included markers towards c.
    π_kernel = PracticalBayes.ConjugateGibbs(:π, :beta, (b, c)) do cc
        δ = cc.values.δ
        nincl = count(==(1), δ)
        (length(δ) - nincl, nincl)
    end

    if variant === :C
        model = _bayes_c(y, X, b, c, S2b, S2e, νb, νe)
        σ²ᵦ_kernel = InvGammaVarKernel(:σ²ᵦ, (νb, S2b)) do cc
            β = cc.values.β
            δ = cc.values.δ
            # Sum of squares over included markers; d.f. = number included.
            ss = 0.0
            k = 0
            @inbounds for j in eachindex(δ)
                if δ[j] == 1
                    ss += β[j]^2
                    k += 1
                end
            end
            (k, ss)
        end
        spl = Gibbs(
            (:β, :δ) => βδ_kernel,
            :σ²ᵦ => σ²ᵦ_kernel,
            :σ²ₑ => σ²ₑ_kernel,
            :π => π_kernel,
        )
    else
        model = _bayes_b(y, X, b, c, S2b, S2e, νb, νe)
        spl = Gibbs(
            (:β, :δ, :τ²) => βδ_kernel,
            :σ²ₑ => σ²ₑ_kernel,
            :π => π_kernel,
        )
    end
    return model, spl
end

# `β`, `δ` (and BayesB's `τ²`) are full-length families with one `~` site per
# marker, so PracticalBayes' Gibbs init builds length-`p` vectors that match what
# `BayesAlphabetKernel` returns. The spike (βⱼ = 0 when δⱼ = 0) is enforced by
# the kernel; the model body provides role discovery, initial values and
# log-density bookkeeping.
@model function _bayes_c(y, X, b, c, S2b, S2e, νb, νe)
    p = size(X, 2)
    σ²ₑ ~ InverseGamma(νe / 2, νe * S2e / 2)
    σ²ᵦ ~ InverseGamma(νb / 2, νb * S2b / 2)
    π ~ Beta(b, c)
    δ = Vector{Int}(undef, p)
    β = Vector{paramtype(__mode__)}(undef, p)
    for j in 1:p
        δ[j] ~ Bernoulli(1.0 - π)
    end
    for j in 1:p
        β[j] ~ Normal(0.0, sqrt(σ²ᵦ))
    end
    y .~ Normal.(X * (β .* δ), sqrt(σ²ₑ))
end

@model function _bayes_b(y, X, b, c, S2b, S2e, νb, νe)
    p = size(X, 2)
    σ²ₑ ~ InverseGamma(νe / 2, νe * S2e / 2)
    π ~ Beta(b, c)
    δ = Vector{Int}(undef, p)
    τ² = Vector{paramtype(__mode__)}(undef, p)
    β = Vector{paramtype(__mode__)}(undef, p)
    for j in 1:p
        δ[j] ~ Bernoulli(1.0 - π)
    end
    for j in 1:p
        τ²[j] ~ InverseGamma(νb / 2, νb * S2b / 2)
    end
    for j in 1:p
        β[j] ~ Normal(0.0, sqrt(τ²[j]))
    end
    y .~ Normal.(X * (β .* δ), sqrt(σ²ₑ))
end
