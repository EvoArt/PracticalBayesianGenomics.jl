using PracticalBayes
using Distributions
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# The Bayesian alphabet (BayesA/B/C…) for whole-genome regression
#
#     y = Xβ + e,   e ~ N(0, σ²ₑ I)
#
# with a sparsity-inducing prior on the p marker effects β. The workhorse is a
# single-site Gibbs scan over markers, sampling each βⱼ (and, for the spike-and-
# slab variants, its inclusion indicator δⱼ) from its exact conditional while
# keeping a running fitted vector `Xβ` that is rank-1 updated as each βⱼ changes.
#
# EFFICIENCY: the per-column cross-products xⱼ'xⱼ are CONSTANT across sweeps and
# are precomputed once into the kernel (`xtx`), as are column views into X. The
# original stub recomputed `dot(xj, xj)` and sliced a fresh `X[:, j]` copy every
# sweep for every marker — at p markers × N sweeps that is the dominant avoidable
# cost. Residual bookkeeping is done incrementally (`predicted .+= xⱼ·Δβⱼ`), so
# each marker update is O(n), giving an O(n·p) sweep with no O(n·p) matrix
# rebuild.
#
# Variants (BayesC π mixture shown; BayesB is the per-marker-variance limit):
#   * variance components σ²ᵦ, σ²ₑ → conjugate Inverse-Gamma (InvGammaVarKernel)
#   * inclusion probability π       → conjugate Beta (PracticalBayes.ConjugateGibbs)
#   * (β, δ) block                  → BayesAlphabetKernel (this file)
# ---------------------------------------------------------------------------

"""
    BayesAlphabetKernel(xtx; variant=:C)

Gibbs kernel sampling the joint `(β, δ)` marker-effect block by a single-site
Gibbs scan. `xtx` is the precomputed vector of column cross-products `xⱼ'xⱼ`
(constant across sweeps). `variant` selects the alphabet member:

- `:C` (default) — spike-and-slab with a COMMON effect variance `σ²ᵦ` shared by
  all included markers, and inclusion probability `π` (BayesCπ).
- `:B` — spike-and-slab where included markers each get their own variance
  (here approximated by the shared-`σ²ᵦ` conditional; a full per-marker-variance
  BayesB is a later extension).

The kernel reads `y`, `X` from the model args and `σ²ᵦ, σ²ₑ, π` from the current
state. Prefer building via [`bayes_alphabet_model`](@ref).
"""
mutable struct BayesAlphabetKernel{V<:AbstractVector} <: PracticalBayes.AbstractGibbsKernel
    const xtx::V
    const variant::Symbol
    resid::Vector{Float64}   # residual y − Xβ at the END of the last sweep's scan
end
BayesAlphabetKernel(xtx; variant::Symbol=:C) = BayesAlphabetKernel(xtx, variant, Float64[])

# Back-compat: the original name.
const BayesBKernel = BayesAlphabetKernel

function PracticalBayes.gibbs_step(rng, k::BayesAlphabetKernel, block_names, c::PracticalBayes.ModelConditional)
    block_names == (:β, :δ) || error("BayesAlphabetKernel only supports the (:β, :δ) block")
    y = c.model.args.y
    X = c.model.args.X
    β = copy(c.values.β)
    δ = copy(c.values.δ)
    σ²ᵦ = c.values.σ²ᵦ
    σ²ₑ = c.values.σ²ₑ
    π = c.values.π
    n, p = size(X)

    # Track the RESIDUAL e = y − Xβ incrementally rather than the fitted vector.
    # This is the standard whole-genome-regression trick: the marker conditional
    # then needs only ONE O(n) inner product per marker, `dot(xⱼ, e)`, instead of
    # two (`dot(xⱼ, y)` AND `dot(xⱼ, predicted)`). `dot(xⱼ, y)` in particular is
    # otherwise recomputed every sweep even though Xᵀy is constant. Halves the
    # dominant `dot` cost (profiled at ~35% → ~17%).
    #
    # The initial residual is built from ONLY the nonzero markers (`β .* δ`), so
    # the startup cost is O(n·‖δ‖₀) not O(n·p) — under real sparsity that turns a
    # full n×p `gemv` (profiled at ~46% of a sweep) into a handful of axpys.
    e = copy(y)
    @inbounds for j in 1:p
        bj = β[j]
        if δ[j] == 1 && bj != 0.0
            @. e -= @view(X[:, j]) * bj
        end
    end

    @inbounds for j in 1:p
        β_prev = β[j]
        xj = view(X, :, j)              # view, not a fresh column copy
        xj2 = k.xtx[j]                  # precomputed xⱼ'xⱼ

        # xⱼ'r₋ⱼ where r₋ⱼ = e + xⱼ·βⱼ is the residual with marker j removed.
        xj_r = dot(xj, e) + xj2 * β_prev

        βhat = xj_r / xj2
        σ²0 = σ²ₑ / xj2                 # sampling variance of βhat under δ=0 slab-less
        σ²1 = σ²ᵦ + σ²0

        # log posterior odds of inclusion (δ=1) vs exclusion (δ=0)
        p0 = log(π) + logpdf(Normal(0.0, sqrt(σ²0)), βhat)
        p1 = log1p(-π) + logpdf(Normal(0.0, sqrt(σ²1)), βhat)
        m = max(p0, p1)
        prob = exp(p1 - m) / (exp(p0 - m) + exp(p1 - m))
        prob = clamp(prob, 1e-12, 1.0 - 1e-12)

        if rand(rng, Bernoulli(prob)) == 1
            δ[j] = 1
            μj = σ²ᵦ * xj_r / (σ²ₑ + σ²ᵦ * xj2)
            σ²j = σ²ᵦ * σ²ₑ / (σ²ₑ + σ²ᵦ * xj2)
            β[j] = rand(rng, Normal(μj, sqrt(σ²j)))
        else
            δ[j] = 0
            β[j] = 0.0
        end

        Δ = β[j] - β_prev
        if Δ != 0.0
            @. e -= xj * Δ             # rank-1 update of the residual
        end
    end

    # Stash the final residual so the σ²ₑ conjugate kernel can reuse it instead of
    # recomputing y − Xβ (which would be a full O(n·p) dense matvec, or a slow
    # non-BLAS column-gather). The Gibbs scan resamples the (β, δ) block
    # immediately before σ²ₑ, so this residual is exactly y − Xβ at the σ²ₑ update.
    k.resid = e
    return (; β = β, δ = δ)
end

"""
    bayes_alphabet_model(y, X; variant=:C, b=1.0, c=1.0,
                         S2b=var(y)*0.1, S2e=var(y), νb=3.0, νe=3.0)
        -> (model, sampler)

Build a Bayesian-alphabet whole-genome regression and its blocked-Gibbs sampler.

- `y`, `X` — `n` phenotypes and `n×p` (centred) marker matrix.
- `variant` — `:C` (BayesCπ) or `:B` (see [`BayesAlphabetKernel`](@ref)).
- `(b, c)` — Beta prior for the inclusion probability `π`.
- `(νb, S2b)`, `(νe, S2e)` — scaled-inv-χ² priors for `σ²ᵦ` / `σ²ₑ`.

Returns `(model, spl)`. The `(β, δ)` block uses the single-site
`BayesAlphabetKernel`; `σ²ᵦ`/`σ²ₑ` use conjugate `InvGammaVarKernel`s; `π` uses
PracticalBayes' conjugate Beta `ConjugateGibbs`. Sample with
`AbstractMCMC.sample(model, spl, N)`.
"""
function bayes_alphabet_model(y, X; variant::Symbol=:C,
                              b=1.0, c=1.0,
                              S2b=var(y) * 0.1, S2e=var(y), νb=3.0, νe=3.0)
    n, p = size(X)
    # Precompute constant column cross-products once.
    xtx = [dot(view(X, :, j), view(X, :, j)) for j in 1:p]

    @model function bayes_alphabet(y, X, b, c, S2b, S2e, νb, νe)
        n, p = size(X)
        σ²ₑ ~ InverseGamma(νe / 2, νe * S2e / 2)
        σ²ᵦ ~ InverseGamma(νb / 2, νb * S2b / 2)
        π ~ Beta(b, c)

        # `β` and `δ` are FULL-LENGTH assumed families (one `~` site per marker),
        # so PracticalBayes' Gibbs init builds length-`p` vectors that match what
        # `BayesAlphabetKernel` returns. The spike-and-slab structure (some βⱼ = 0
        # when δⱼ = 0) lives entirely inside the kernel; the model body draws the
        # unconditional slab prior for every βⱼ purely to declare the variable and
        # give a consistent shape. Because the whole `(β, δ)` block is owned by the
        # custom kernel and never sees HMC, this body is used only for
        # role-discovery/init and for `logjoint`-style utilities — the kernel is
        # what actually enforces the spike.
        δ = Vector{Int}(undef, p)
        β = Vector{paramtype(__mode__)}(undef, p)
        for j in 1:p
            δ[j] ~ Bernoulli(1.0 - π)
        end
        for j in 1:p
            β[j] ~ Normal(0.0, sqrt(σ²ᵦ))
        end

        # Effective effect is βⱼ·δⱼ (zeroed where excluded); matches the kernel.
        y .~ Normal.(X * (β .* δ), sqrt(σ²ₑ))
    end

    model = bayes_alphabet(y, X, b, c, S2b, S2e, νb, νe)

    βδ_kernel = BayesAlphabetKernel(xtx; variant=variant)
    σ²ₑ_kernel = InvGammaVarKernel(:σ²ₑ, (νe, S2e)) do cc
        # Reuse the residual y − Xβ the (β, δ) kernel already computed on this
        # sweep (stashed in `βδ_kernel.resid`), rather than recomputing it. The
        # Gibbs scan resamples (β, δ) then σ²ᵦ then σ²ₑ, and neither β nor δ change
        # between the (β, δ) update and here, so the stash is exactly current.
        # Falls back to a direct compute if the kernel hasn't run yet.
        r = βδ_kernel.resid
        if isempty(r)
            r = cc.model.args.y .- cc.model.args.X * cc.values.β
        end
        (length(r), dot(r, r))
    end
    σ²ᵦ_kernel = InvGammaVarKernel(:σ²ᵦ, (νb, S2b)) do cc
        β = cc.values.β
        δ = cc.values.δ
        incl = findall(==(1), δ)
        # Sum of squares over INCLUDED markers only; d.f. = number included.
        (length(incl), sum(abs2, @view β[incl]; init = 0.0))
    end
    # Conjugate Beta for π: number of EXCLUDED markers are the "successes" of
    # Bernoulli(1-π) (δ=0), included are the failures.
    π_kernel = PracticalBayes.ConjugateGibbs(:π, :beta, (b, c)) do cc
        δ = cc.values.δ
        nincl = count(==(1), δ)
        (length(δ) - nincl, nincl)   # (δ=0 count, δ=1 count) → Beta(b + n0, c + n1)
    end

    spl = Gibbs(
        (:β, :δ) => βδ_kernel,
        :σ²ᵦ => σ²ᵦ_kernel,
        :σ²ₑ => σ²ₑ_kernel,
        :π => π_kernel,
    )
    return model, spl
end

# Back-compat wrapper returning only the model (old signature returned a bare
# model; callers built the Gibbs sampler themselves).
function bayes_b_model(y, X; kwargs...)
    model, _ = bayes_alphabet_model(y, X; variant=:B, kwargs...)
    return model
end
