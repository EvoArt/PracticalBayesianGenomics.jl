using PracticalBayes
using Distributions
using LinearAlgebra
using SparseArrays
using SparseArrays: CHOLMOD
using Random

# ---------------------------------------------------------------------------
# The Gaussian animal model
#
#     y = Xβ + Zu + e,   u ~ N(0, σ²ᵤ K),   e ~ N(0, σ²ₑ I)
#
# with K the relatedness matrix (pedigree A, genomic G, or single-step H — see
# relatedness.jl). Sampled by a blocked Gibbs scheme in which every block has a
# closed-form conditional:
#
#   * u        — the breeding values: exact multivariate-normal conditional,
#                sampled by `AnimalKernel` (below).
#   * β        — fixed effects: a flat / broad-normal prior, so its conditional
#                is also Gaussian. Left on NUTS by default (small p; cheap), but
#                can be given a conjugate kernel too.
#   * σ²ᵤ, σ²ₑ — variance components: conjugate Inverse-Gamma, via
#                `InvGammaVarKernel` (variance_kernels.jl).
#
# EFFICIENCY: the relatedness structure (`K⁻¹`) is precomputed ONCE and carried
# inside `AnimalKernel`. Each sweep rebuilds only the precision
# `Z'Z/σ²ₑ + K⁻¹/σ²ᵤ` (the two scalar variances are all that change) and
# factorizes it once. `K` is never re-inverted per sweep — the original stub's
# per-sweep `inv(A)` + fresh dense Cholesky was the dominant cost at q = 2500.
# `Z'Z` and `Z'` are also cached, since they never change.
# ---------------------------------------------------------------------------

"""
    AnimalKernel(rel, ZtZ, Zt)

Gibbs kernel that samples the breeding-value block `u` from its exact
multivariate-normal full conditional

    u | rest ~ N(P⁻¹ b, P⁻¹),   P = Z'Z/σ²ₑ + K⁻¹/σ²ᵤ,   b = Z'(y − Xβ)/σ²ₑ.

`rel::RelatednessStructure` carries the precomputed `K⁻¹`; `ZtZ = Z'Z` and
`Zt = Z'` are cached at build time. Per sweep only the precision `P` (which
depends on the two current variances) is rebuilt and factorized once.

The kernel is **mutable** and caches the Cholesky *symbolic* factorization
across sweeps. The precision's SPARSITY PATTERN is constant (only the two scalar
variances change), so the expensive CHOLMOD symbolic-analysis / fill-reducing-
ordering step (`cholmod_l_analyze`, profiled at ~30% of a sweep) is done ONCE
and reused via in-place `cholesky!` every sweep afterwards. The precision is
kept DENSE for a dense (genomic `G⁻¹`) `K⁻¹` and SPARSE for a sparse (pedigree
`A⁻¹`) one — the original broadcast `sparse ./ s .+ dense ./ s` silently
returned a *sparse* matrix even for the dense genomic case, forcing the whole
dense precision through the sparse CHOLMOD path (fill-in + re-analysis every
sweep); dispatching on `K⁻¹`'s storage fixes that.

Prefer building this via [`animal_model`](@ref) rather than by hand.
"""
mutable struct AnimalKernel{R<:RelatednessStructure,MZ<:AbstractMatrix,MT<:AbstractMatrix} <: PracticalBayes.AbstractGibbsKernel
    const rel::R
    const ZtZ::MZ
    const Zt::MT
    factor::Any        # cached Cholesky factor (dense Cholesky or CHOLMOD Factor), or nothing
end
AnimalKernel(rel, ZtZ, Zt) = AnimalKernel(rel, ZtZ, Zt, nothing)

function PracticalBayes.gibbs_step(rng, k::AnimalKernel, block_names, c::PracticalBayes.ModelConditional)
    block_names == (:u,) || error("AnimalKernel only supports the :u block")
    y = c.model.args.y
    X = c.model.args.X
    β = c.values.β
    σ²ᵤ = c.values.σ²ᵤ
    σ²ₑ = c.values.σ²ₑ

    resid = y .- X * β
    b = (k.Zt * resid) ./ σ²ₑ
    P = _precision(k.ZtZ, k.rel.Kinv, σ²ₑ, σ²ᵤ)
    u, k.factor = _draw_gmrf(rng, P, b, k.factor)
    return (; u = u)
end

# Build P = Z'Z/σ²ₑ + K⁻¹/σ²ᵤ with the storage type dictated by K⁻¹: dense when
# K⁻¹ is dense (genomic G⁻¹ → fast LAPACK Cholesky), sparse when K⁻¹ is sparse
# (pedigree A⁻¹ → CHOLMOD, symbolic factor reused). Written so that neither case
# ever materialises the "wrong" storage class.
_precision(ZtZ, Kinv::AbstractMatrix, σ²ₑ, σ²ᵤ) =
    (Matrix(ZtZ) ./ σ²ₑ) .+ (Kinv ./ σ²ᵤ)                    # dense genomic case
_precision(ZtZ::SparseMatrixCSC, Kinv::SparseMatrixCSC, σ²ₑ, σ²ᵤ) =
    (ZtZ ./ σ²ₑ) .+ (Kinv ./ σ²ᵤ)                            # sparse pedigree case

# Draw x ~ N(P⁻¹ b, P⁻¹) via one Cholesky, reusing a cached factor.
# Dense P: LAPACK Cholesky. P = UᵀU, so U \ z ~ N(0, P⁻¹) supplies the noise.
# There is no cheap in-place symbolic reuse for dense LAPACK, so we just
# refactorize (dense Cholesky is already O(q³) with no separate analysis phase).
function _draw_gmrf(rng, P::Matrix, b::AbstractVector, _factor)
    q = length(b)
    F = cholesky!(Symmetric(P))          # in-place: P is a fresh per-sweep matrix
    μ = F \ b
    z = randn(rng, q)
    return μ .+ (F.U \ z), F
end

# Sparse P (pedigree): reuse the cached CHOLMOD symbolic factor. The first call
# runs the full `cholesky` (symbolic analysis + numeric factorization); every
# later call runs `cholesky!(F, P)`, which reuses F's ordering/symbolic structure
# and only redoes the O(nnz) numeric factorization — eliminating the per-sweep
# `cholmod_l_analyze`.
function _draw_gmrf(rng, P::SparseMatrixCSC, b::AbstractVector, factor)
    S = Symmetric(P)
    if factor === nothing
        F = cholesky(S)
    else
        F = factor
        cholesky!(F, S)                  # refactorize in place, reuse symbolic factor
    end
    μ = F \ b
    q = length(b)
    # Sample the N(0, P⁻¹) noise through CHOLMOD's own triangular/permutation
    # solves rather than extracting `sparse(F.L)` (which cost ~15% of a sweep in
    # `cholmod_l_factor_to_sparse`/`copy_factor`). CHOLMOD factorizes the PERMUTED
    # matrix P[p,p] = L·Lᵀ, so with system codes CHOLMOD_Lt (=5) then CHOLMOD_Pt
    # (=8): Pᵀ·(Lᵀ \ z) has covariance P⁻¹. Verified against a dense reference
    # sampler (cov rel-err ≈ 5e-3 at N = 2e5).
    z = CHOLMOD.Dense(randn(rng, q))
    w = CHOLMOD.solve(CHOLMOD.CHOLMOD_Pt, F, CHOLMOD.solve(CHOLMOD.CHOLMOD_Lt, F, z))
    return μ .+ vec(Vector(w)), F
end

"""
    GaussianEffectKernel(name, XtX, Xt, Zt, prior_var)

Gibbs kernel that samples a Gaussian fixed-effect block `name` (here `β`) from
its conjugate multivariate-normal full conditional under a `N(0, prior_var·I)`
prior. `XtX = X'X`, `Xt = X'`, and the cached sparse `Zt = Z'` are supplied at
build time. The conditional precision is `X'X/σ²ₑ + I/prior_var`; the conditional
mean solves that against `X'(y − Zu)/σ²ₑ`, with `Zu` done via the cached SPARSE
`Zt`. For the small `p` typical of fixed effects this is a tiny dense solve, done
in closed form so `β` needs no NUTS/AD step.
"""
struct GaussianEffectKernel{MX<:AbstractMatrix,MT<:AbstractMatrix,MZ<:AbstractMatrix} <: PracticalBayes.AbstractGibbsKernel
    name::Symbol
    XtX::MX
    Xt::MT
    Zt::MZ
    prior_var::Float64
end
GaussianEffectKernel(name::Symbol, XtX, Xt, Zt, prior_var) =
    GaussianEffectKernel(name, XtX, Xt, Zt, Float64(prior_var))

function PracticalBayes.gibbs_step(rng, k::GaussianEffectKernel, block_names, c::PracticalBayes.ModelConditional)
    block_names == (k.name,) || error("GaussianEffectKernel owns block $(k.name), got $(block_names)")
    y = c.model.args.y
    u = c.values.u
    σ²ₑ = c.values.σ²ₑ
    p = size(k.XtX, 1)
    target = y .- k.Zt' * u                  # cached sparse Zᵀ transposed = Z (sparse)
    b = (k.Xt * target) ./ σ²ₑ
    P = Matrix(k.XtX) ./ σ²ₑ .+ Matrix{Float64}(I, p, p) ./ k.prior_var
    draw, _ = _draw_gmrf(rng, P, b, nothing)   # p is tiny; no factor caching needed
    return NamedTuple{(k.name,)}((draw,))
end

"""
    animal_model(y, X, Z, rel; β_scale=1e6, νu=3.0, Su2=1.0, νe=3.0, Se2=1.0)
        -> (model, sampler)

Build the animal model and a ready-to-use blocked-Gibbs sampler for it.

- `y`      — `n`-vector of phenotypes.
- `X`      — `n×p` fixed-effect design (intercept/covariates).
- `Z`      — `n×q` incidence matrix mapping records to the `q` animals in `rel`.
- `rel`    — a [`RelatednessStructure`](@ref) (`pedigree_relatedness`,
             `genomic_relatedness`, or `singlestep_relatedness`). Its `K⁻¹` is
             precomputed once; the sampler reuses it every sweep.
- `β_scale`— prior variance for the flat fixed effects.
- `(νu, Su2)`, `(νe, Se2)` — scaled-inv-χ² prior d.f./scale for σ²ᵤ / σ²ₑ.

Returns `(model, spl)`. `spl` samples `u` with the exact MVN `AnimalKernel`,
`σ²ᵤ`/`σ²ₑ` with conjugate `InvGammaVarKernel`s, and `β` with NUTS (cheap for
small `p`). Sample with `AbstractMCMC.sample(model, spl, N)` /
`AbstractMCMC.step`.

The relatedness matrix `K` is written into the model body ONLY as the prior
`u ~ MvNormal(0, σ²ᵤ K)` used for coverage/log-density bookkeeping; the actual
`u` update never uses `K` (only the cached `K⁻¹`). For the pedigree case, where
`K = A` is expensive to form densely, pass the model a dense `K` only if you need
log-joint evaluation — otherwise the sampler runs entirely off `K⁻¹`.
"""
function animal_model(y, X, Z, rel::RelatednessStructure;
                      β_scale=1e6, νu=3.0, Su2=1.0, νe=3.0, Se2=1.0)
    q = size(Z, 2)
    size(rel, 1) == q || throw(DimensionMismatch("rel is $(size(rel,1))×$(size(rel,1)) but Z has $q columns"))

    # Precompute the design products once, all in SPARSE storage. `Z` is an
    # incidence matrix (mostly zeros); keeping every `Z`/`Zᵀ` multiply sparse is
    # essential — a dense q×q `Z·u` was profiled at ~33% of a pedigree sweep,
    # dwarfing the sparse Cholesky itself. `Zsp` is what every kernel multiplies
    # by, regardless of how the caller passed `Z`.
    Zsp = sparse(Z)
    Zt = sparse(transpose(Zsp))
    ZtZ = Zt * Zsp

    @model function animal_model_internal(y, X, Z, Kinv, β_scale, νu, Su2, νe, Se2)
        p = size(X, 2)
        q = size(Z, 2)

        β ~ MvNormal(zeros(paramtype(__mode__), p), β_scale * I(p))
        σ²ᵤ ~ InverseGamma(νu / 2, νu * Su2 / 2)
        σ²ₑ ~ InverseGamma(νe / 2, νe * Se2 / 2)

        # Prior on u expressed through K⁻¹ (precision form): logpdf uses the
        # cached inverse directly, avoiding any dense K in the model body.
        u ~ MvNormalCanon(zeros(paramtype(__mode__), q), Symmetric(Matrix(Kinv)) ./ σ²ᵤ)

        y .~ Normal.(X * β .+ Z * u, sqrt(σ²ₑ))
    end

    model = animal_model_internal(y, X, Z, rel.Kinv, β_scale, νu, Su2, νe, Se2)

    # Cache the fixed-effect design products too (constant across sweeps).
    XtX = Symmetric(Matrix(X' * X))

    ukernel = AnimalKernel(rel, ZtZ, Zt)
    βkernel = GaussianEffectKernel(:β, XtX, Matrix(X'), Zt, β_scale)
    Kinv = rel.Kinv
    σ²ᵤ_kernel = InvGammaVarKernel(:σ²ᵤ, (νu, Su2)) do c
        u = c.values.u
        # Random-effect sum of squares in the K-metric: u' K⁻¹ u (sparse matvec
        # for pedigree A⁻¹). Captured `Kinv`, not re-fetched from model args.
        (length(u), dot(u, Kinv * u))
    end
    σ²ₑ_kernel = InvGammaVarKernel(:σ²ₑ, (νe, Se2)) do c
        # resid = y − Xβ − Zu, with Zu done via the cached SPARSE Zᵀ transposed.
        resid = c.model.args.y .- c.model.args.X * c.values.β .- Zt' * c.values.u
        (length(resid), dot(resid, resid))
    end

    # Fully-conjugate blocked Gibbs — no NUTS anywhere, so no AD/HMC cost and no
    # AdvancedHMC dependency. Every block has a closed-form conditional.
    spl = Gibbs(
        :β => βkernel,
        :u => ukernel,
        :σ²ᵤ => σ²ᵤ_kernel,
        :σ²ₑ => σ²ₑ_kernel,
    )
    return model, spl
end
