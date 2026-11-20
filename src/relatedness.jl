using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Relatedness structures for the animal model.
#
# The animal model's random effect `u` (breeding values) has a full conditional
#
#     u | rest  ~  N(P⁻¹ b,  P⁻¹),   P = Z'Z/σ²ₑ + K⁻¹/σ²ᵤ,   b = Z'(y - Xβ)/σ²ₑ
#
# where `K` is the relatedness matrix (pedigree `A`, genomic `G`, or the single-
# step `H`). The ONLY thing that varies with the choice of pedigree vs. genomic
# is how `K⁻¹` is obtained — and, crucially, `K⁻¹` does not change across Gibbs
# sweeps. Only the two scalar variances do. So the whole "form K⁻¹" cost must be
# paid ONCE, at model-build time, not every sweep.
#
# This is the efficiency bug in the original stub: it called `inv(Symmetric(A))`
# and a fresh dense Cholesky of the q×q precision EVERY sweep — an O(q³) inverse
# plus an O(q³) factorization per iteration at q = 2500. Here `K⁻¹` is
# precomputed and carried inside the kernel struct; each sweep only rebuilds the
# precision from the cached `K⁻¹` and the two current variances and factorizes
# once (unavoidable, since the variances change), never re-inverting K.
#
# A `RelatednessStructure` is exactly "a precomputed K⁻¹ plus the label of where
# it came from." The `u`-block kernel (animal_model.jl) is written against this
# abstraction, so the same sampler serves pedigree A, genomic G, and single-step
# H without change — only the constructor differs.
# ---------------------------------------------------------------------------

"""
    RelatednessStructure

Holds a precomputed inverse relatedness matrix `Kinv` (`K⁻¹`) and a `kind` label
(`:pedigree`, `:genomic`, or `:singlestep`). Built ONCE via one of
[`pedigree_relatedness`](@ref), [`genomic_relatedness`](@ref), or
[`singlestep_relatedness`](@ref); reused every Gibbs sweep by the animal model's
`u`-block kernel, which never re-inverts `K`.

`Kinv` may be sparse (pedigree `A⁻¹` from Henderson's rules) or dense (genomic
`G⁻¹`); the kernel handles both — `Z'Z/σ²ₑ + Kinv/σ²ᵤ` stays sparse iff both
terms are, and `cholesky` dispatches to the sparse or dense factorization
accordingly.
"""
struct RelatednessStructure{M<:AbstractMatrix}
    Kinv::M
    kind::Symbol
end

Base.size(r::RelatednessStructure, args...) = size(r.Kinv, args...)

# ---------------------------------------------------------------------------
# Pedigree A⁻¹ via Henderson's rules — never forms dense A.
# ---------------------------------------------------------------------------

"""
    pedigree_relatedness(sire, dam) -> RelatednessStructure

Build the sparse inverse numerator-relationship matrix `A⁻¹` directly from a
pedigree, via Henderson's (1976) rules, WITHOUT ever forming the dense `A`. For
a pedigree of `q` animals this is O(q) work and O(q) nonzeros — the reason the
classic animal model scales to millions of animals where a dense `A` would not.

`sire` and `dam` are length-`q` integer vectors: `sire[i]` / `dam[i]` are the
row indices (in `1:q`) of animal `i`'s parents, or `0` for unknown. Animals must
be ordered so that parents precede offspring (a topological sort of the
pedigree) — the standard requirement; `sire[i] < i` and `dam[i] < i` (or `0`)
for every `i`.

Returns a [`RelatednessStructure`](@ref) wrapping the sparse `A⁻¹`.

Henderson's rules: with inbreeding-adjusted Mendelian variances
`dᵢ = 1 / (½ - ¼(Fₛ + F_d))` (using `Fₛ, F_d` the parents' inbreeding
coefficients, computed here via the recursive method), each animal contributes
`dᵢ` to `A⁻¹[i,i]`, `-dᵢ/2` to the `(i, parent)` cells, and `dᵢ/4` to the
`(parent, parent')` cells for each present parent.
"""
function pedigree_relatedness(sire::AbstractVector{<:Integer}, dam::AbstractVector{<:Integer})
    q = length(sire)
    length(dam) == q || throw(DimensionMismatch("sire and dam must have equal length"))
    for i in 1:q
        (sire[i] == 0 || sire[i] < i) || throw(ArgumentError("pedigree not sorted: sire[$i]=$(sire[i]) ≥ $i"))
        (dam[i] == 0 || dam[i] < i) || throw(ArgumentError("pedigree not sorted: dam[$i]=$(dam[i]) ≥ $i"))
    end

    F = inbreeding_coefficients(sire, dam)

    # Accumulate A⁻¹ into COO triplets, then let sparse assembly sum duplicates.
    I = Int[]; J = Int[]; V = Float64[]
    push_cell!(i, j, v) = (push!(I, i); push!(J, j); push!(V, v))
    for i in 1:q
        s, d = sire[i], dam[i]
        Fs = s == 0 ? 0.0 : F[s]
        Fd = d == 0 ? 0.0 : F[d]
        # Mendelian sampling variance, adjusted for parents' inbreeding.
        denom = 0.5 - 0.25 * (Fs + Fd)  # both parents known
        s == 0 && (denom = 0.75 - 0.25 * Fd)  # one parent unknown
        d == 0 && (denom = 0.75 - 0.25 * Fs)
        (s == 0 && d == 0) && (denom = 1.0)   # both unknown
        di = 1.0 / denom

        push_cell!(i, i, di)
        if s != 0
            push_cell!(i, s, -di / 2); push_cell!(s, i, -di / 2)
            push_cell!(s, s, di / 4)
        end
        if d != 0
            push_cell!(i, d, -di / 2); push_cell!(d, i, -di / 2)
            push_cell!(d, d, di / 4)
        end
        if s != 0 && d != 0
            push_cell!(s, d, di / 4); push_cell!(d, s, di / 4)
        end
    end
    Ainv = sparse(I, J, V, q, q)  # sparse() sums duplicate (i,j) contributions
    return RelatednessStructure(Ainv, :pedigree)
end

"""
    inbreeding_coefficients(sire, dam) -> Vector{Float64}

Inbreeding coefficient `Fᵢ` of each animal, computed by the recursive tabular
method (Fᵢ = ½·A[sire, dam], the numerator relationship between its parents).
Only the relationship entries actually needed are computed, memoised in a small
`Dict`, so this is far cheaper than forming the whole `A`. Parents must precede
offspring (see [`pedigree_relatedness`](@ref)).
"""
function inbreeding_coefficients(sire::AbstractVector{<:Integer}, dam::AbstractVector{<:Integer})
    q = length(sire)
    F = zeros(Float64, q)
    memo = Dict{Tuple{Int,Int},Float64}()  # numerator-relationship a(i,j), i ≤ j, memoised

    function a(i::Int, j::Int)
        (i == 0 || j == 0) && return 0.0
        i > j && ((i, j) = (j, i))          # keep i ≤ j
        haskey(memo, (i, j)) && return memo[(i, j)]
        if i == j
            val = 1.0 + 0.5 * a(sire[i], dam[i])   # a(i,i) = 1 + Fᵢ
        else
            # a(i,j) = ½(a(i, sire_j) + a(i, dam_j)), expanding the LATER animal j.
            val = 0.5 * (a(i, sire[j]) + a(i, dam[j]))
        end
        memo[(i, j)] = val
        return val
    end

    for i in 1:q
        F[i] = (sire[i] == 0 || dam[i] == 0) ? 0.0 : 0.5 * a(sire[i], dam[i])
        a(i, i)  # warm the diagonal so later a(i,i) lookups are cheap
    end
    return F
end

# ---------------------------------------------------------------------------
# Genomic G⁻¹ (dense) — GBLUP.
# ---------------------------------------------------------------------------

"""
    genomic_relatedness(G; blend=0.0, base=I) -> RelatednessStructure

Build a dense genomic relatedness structure from a genomic relationship matrix
`G` (e.g. VanRaden's `G = MMᵀ / (2·Σpⱼ(1-pⱼ))` for centred marker matrix `M`).
`G⁻¹` is formed once here via a single Cholesky and reused every sweep.

`blend > 0` returns `(1-blend)·G + blend·base` before inverting — the standard
regularisation that keeps `G` invertible when markers make it singular/near-
singular (`base` defaults to the identity, i.e. a ridge toward independence; pass
a pedigree `A` restricted to the genotyped animals to blend toward pedigree
instead). Use a small `blend` (e.g. 0.01–0.05) on real data.
"""
function genomic_relatedness(G::AbstractMatrix; blend::Real=0.0, base=I)
    q = size(G, 1)
    size(G, 2) == q || throw(DimensionMismatch("G must be square"))
    if blend > 0
        base_mat = base isa UniformScaling ? Matrix{Float64}(base, q, q) : Matrix{Float64}(base)
        Gb = (1 - blend) .* Matrix{Float64}(G) .+ blend .* base_mat
    else
        Gb = Matrix{Float64}(G)
    end
    Ginv = inv(cholesky(Symmetric(Gb)))
    return RelatednessStructure(Matrix(Ginv), :genomic)
end

"""
    genomic_relatedness_from_markers(M; centered=false, blend=0.01) -> RelatednessStructure

Convenience: build `G` from a raw `n×p` marker dosage matrix `M` (entries in
`{0,1,2}`) using VanRaden's method 1, then delegate to [`genomic_relatedness`](@ref).
Columns are centred by `2pⱼ` (allele frequencies estimated from `M` unless
`centered=true`, meaning `M` is already centred) and `G = ZZᵀ / (2·Σpⱼ(1-pⱼ))`.
"""
function genomic_relatedness_from_markers(M::AbstractMatrix; centered::Bool=false, blend::Real=0.01)
    if centered
        Z = Matrix{Float64}(M)
        p = fill(0.5, size(M, 2))  # scaling below only needs Σp(1-p); use column-implied
    else
        p = vec(sum(M; dims=1)) ./ (2 * size(M, 1))
        Z = Matrix{Float64}(M) .- 2 .* p'
    end
    scale = 2 * sum(pj -> pj * (1 - pj), p)
    G = (Z * Z') ./ scale
    return genomic_relatedness(G; blend=blend)
end

# ---------------------------------------------------------------------------
# Single-step H⁻¹ (ssGBLUP) — combines pedigree and genomic.
# ---------------------------------------------------------------------------

"""
    singlestep_relatedness(sire, dam, G, genotyped; blend=0.05, τ=1.0, ω=1.0)
        -> RelatednessStructure

Single-step (ssGBLUP) inverse `H⁻¹`, combining a full pedigree with a genomic
relationship matrix on the subset of `genotyped` animals. Uses Aguilar/Legarra's
closed form

    H⁻¹ = A⁻¹ + [0 0; 0  τ·G⁻¹ − ω·A₂₂⁻¹]

where block `2` is the genotyped animals and `A₂₂` is `A` restricted to them.
`G` is blended toward `A₂₂` (`blend`) for invertibility. `τ`, `ω` are the usual
tuning weights (both 1 by default). `genotyped` is the vector of row indices (in
the pedigree ordering) of the animals in `G`, in the same order as `G`'s rows.

Requires the pedigree submatrix `A₂₂`, obtained by inverting the corresponding
block of `A⁻¹` (small: only the genotyped animals).
"""
function singlestep_relatedness(sire, dam, G::AbstractMatrix, genotyped::AbstractVector{<:Integer};
                                blend::Real=0.05, τ::Real=1.0, ω::Real=1.0)
    ped = pedigree_relatedness(sire, dam)
    Ainv = ped.Kinv
    q = size(Ainv, 1)
    g = collect(genotyped)
    size(G, 1) == length(g) || throw(DimensionMismatch("G size must match number of genotyped animals"))

    # A₂₂ = inverse of the (2,2) block of A ... but we only have A⁻¹. Use the
    # identity A₂₂⁻¹ = (A⁻¹)₂₂ − (A⁻¹)₂₁ (A⁻¹)₁₁⁻¹ (A⁻¹)₁₂ (Schur complement),
    # so A₂₂ needs A₂₂⁻¹ inverted. For clarity/robustness we form A₂₂ directly
    # by solving against A⁻¹'s blocks.
    ng = setdiff(1:q, g)
    A11 = Ainv[ng, ng]; A12 = Ainv[ng, g]; A22 = Ainv[g, g]
    A22inv = Matrix(A22) - Matrix(A12)' * (Matrix(A11) \ Matrix(A12))  # = (A₂₂)⁻¹ Schur complement

    Gb = (1 - blend) .* Matrix(G) .+ blend .* inv(A22inv)  # blend toward A₂₂ = inv(A22inv)
    Ginv = inv(cholesky(Symmetric(Gb)))

    Hinv = Matrix{Float64}(Ainv)
    Hinv[g, g] .+= τ .* Ginv .- ω .* A22inv
    return RelatednessStructure(Hinv, :singlestep)
end
