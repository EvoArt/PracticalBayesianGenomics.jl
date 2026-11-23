using Test
using PracticalBayes
using StableRNGs
using Distributions
using LinearAlgebra
using SparseArrays
using Statistics
using Random: randperm
using PracticalBayesianGenomics
import AbstractMCMC

# Dense tabular numerator-relationship matrix A (reference for pedigree tests).
# Parents must precede offspring.
function dense_A(sire, dam)
    q = length(sire)
    A = zeros(q, q)
    for i in 1:q
        s, d = sire[i], dam[i]
        for j in 1:i-1
            asj = s == 0 ? 0.0 : A[s, j]
            adj = d == 0 ? 0.0 : A[d, j]
            A[i, j] = A[j, i] = 0.5 * (asj + adj)
        end
        asd = (s == 0 || d == 0) ? 0.0 : A[s, d]
        A[i, i] = 1.0 + 0.5 * asd
    end
    return A
end

@testset "PracticalBayesianGenomics" begin

    @testset "Pedigree A⁻¹ (Henderson) vs dense reference" begin
        # no inbreeding
        s1 = [0, 0, 1, 1, 3]; d1 = [0, 0, 2, 2, 4]
        # inbred (5,6 from 3,4 which share both parents)
        s2 = [0, 0, 1, 1, 3, 3]; d2 = [0, 0, 2, 2, 4, 4]
        for (s, d) in ((s1, d1), (s2, d2))
            A = dense_A(s, d)
            rel = pedigree_relatedness(s, d)
            @test Matrix(rel.Kinv) ≈ inv(A) atol = 1e-8
            F = inbreeding_coefficients(s, d)
            @test F ≈ [A[i, i] - 1.0 for i in eachindex(s)] atol = 1e-10
            @test issparse(rel.Kinv)
        end

        # larger random valid pedigree
        rng = StableRNG(7)
        q = 120
        sire = zeros(Int, q); dam = zeros(Int, q)
        for i in 3:q
            sire[i] = rand(rng, 1:i-1); dam[i] = rand(rng, 1:i-1)
            while dam[i] == sire[i]; dam[i] = rand(rng, 1:i-1); end
        end
        rel = pedigree_relatedness(sire, dam)
        @test Matrix(rel.Kinv) ≈ inv(dense_A(sire, dam)) rtol = 1e-8
    end

    @testset "Genomic relatedness build" begin
        rng = StableRNG(3)
        n, m = 60, 200
        M = Float64.(rand(rng, 0:2, n, m))
        rel = genomic_relatedness_from_markers(M; blend=0.05)
        @test rel.kind == :genomic
        @test size(rel) == (n, n)
        # Kinv is a proper inverse: Kinv * G ≈ I
        G = inv(Symmetric(Matrix(rel.Kinv)))
        @test Matrix(rel.Kinv) * G ≈ I atol = 1e-6
    end

    @testset "InvGammaVarKernel conjugate draw" begin
        # Standalone: residual-variance conditional recovers a known σ² on data
        # simulated from N(0, σ²) with a weak prior.
        rng = StableRNG(1)
        σ²_true = 2.0
        r = sqrt(σ²_true) .* randn(rng, 5000)
        k = InvGammaVarKernel(:σ², (1.0, 1.0)) do c
            (length(c.model.args.r), dot(c.model.args.r, c.model.args.r))
        end
        # Fake a ModelConditional-like call by hand via gibbs_step's contract.
        # Build a tiny model exposing `r`.
        @model function varmod(r)
            σ² ~ InverseGamma(0.5, 0.5)
            r .~ Normal.(0.0, sqrt(σ²))
        end
        m = varmod(r)
        c = PracticalBayes.ModelConditional(m, (; σ² = 1.0))
        draws = [PracticalBayes.gibbs_step(rng, k, (:σ²,), c).σ² for _ in 1:2000]
        @test abs(mean(draws) - σ²_true) < 0.15
    end

    @testset "Animal model (genomic) recovers variances" begin
        rng = StableRNG(20260718)
        n = 250
        M = Float64.(rand(rng, 0:2, n, 600))
        rel = genomic_relatedness_from_markers(M; blend=0.03)
        G = inv(Symmetric(Matrix(rel.Kinv)))
        σ²ᵤ, σ²ₑ = 1.0, 0.5
        β_true = [2.0, -1.5]
        u_true = cholesky(Symmetric(σ²ᵤ .* G)).L * randn(rng, n)
        X = hcat(ones(n), randn(rng, n))
        Z = sparse(1.0I, n, n)
        y = X * β_true + u_true + sqrt(σ²ₑ) * randn(rng, n)

        model, spl = animal_model(y, X, Z, rel; νu=1.0, Su2=1.0, νe=1.0, Se2=1.0)
        t, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:400; t, state = AbstractMCMC.step(rng, model, spl, state); end
        σu = Float64[]; σe = Float64[]; βs = Vector{Float64}[]; uacc = zeros(n); cnt = 0
        for _ in 1:800
            t, state = AbstractMCMC.step(rng, model, spl, state)
            push!(σu, t.σ²ᵤ); push!(σe, t.σ²ₑ); push!(βs, copy(t.β))
            uacc .+= t.u; cnt += 1
        end
        @test abs(mean(σu) - σ²ᵤ) < 0.5
        @test abs(mean(σe) - σ²ₑ) < 0.3
        @test maximum(abs.(mean(βs) - β_true)) < 0.4
        @test cor(uacc ./ cnt, u_true) > 0.6
    end

    @testset "Animal model (pedigree) runs" begin
        rng = StableRNG(99)
        q = 150
        sire = zeros(Int, q); dam = zeros(Int, q)
        for i in 3:q
            sire[i] = rand(rng, 1:i-1); dam[i] = rand(rng, 1:i-1)
            while dam[i] == sire[i]; dam[i] = rand(rng, 1:i-1); end
        end
        rel = pedigree_relatedness(sire, dam)
        X = hcat(ones(q), randn(rng, q))
        Z = sparse(1.0I, q, q)
        y = X * [1.0, 0.5] + randn(rng, q)
        model, spl = animal_model(y, X, Z, rel)
        t, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:20; t, state = AbstractMCMC.step(rng, model, spl, state); end
        @test length(t.u) == q
        @test isfinite(t.σ²ᵤ) && isfinite(t.σ²ₑ)
        @test all(isfinite, t.β)
    end

    @testset "BayesAlphabet recovers sparse effects" begin
        rng = StableRNG(11)
        n, p = 300, 400
        X = randn(rng, n, p); X .-= mean(X; dims=1)
        true_idx = randperm(rng, p)[1:8]
        β_true = zeros(p); β_true[true_idx] .= randn(rng, 8) .* 1.5
        y = X * β_true + randn(rng, n)

        model, spl = bayes_alphabet_model(y, X; variant=:C)
        t, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:500; t, state = AbstractMCMC.step(rng, model, spl, state); end
        δacc = zeros(p); βacc = zeros(p); cnt = 0
        for _ in 1:1500
            t, state = AbstractMCMC.step(rng, model, spl, state)
            δacc .+= t.δ; βacc .+= t.β; cnt += 1
        end
        pip = δacc ./ cnt
        @test length(t.β) == p
        @test length(t.δ) == p
        @test all(d == 0 || d == 1 for d in t.δ)
        # true markers should dominate the posterior inclusion probabilities
        top = sortperm(pip; rev=true)[1:8]
        @test length(intersect(top, true_idx)) >= 6
        @test cor(βacc ./ cnt, β_true) > 0.6
    end

    @testset "BayesB: per-marker variances" begin
        rng = StableRNG(11)
        n, p = 300, 400
        X = randn(rng, n, p); X .-= mean(X; dims=1)
        true_idx = randperm(rng, p)[1:8]
        β_true = zeros(p); β_true[true_idx] .= randn(rng, 8) .* 1.5
        y = X * β_true + randn(rng, n)

        model, spl = bayes_alphabet_model(y, X; variant=:B)
        S2b = spl.blocks[1].kernel.S2b
        t, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:500; t, state = AbstractMCMC.step(rng, model, spl, state); end
        δacc = zeros(p); βacc = zeros(p); cnt = 0
        τ²_excluded = Float64[]
        for _ in 1:1500
            t, state = AbstractMCMC.step(rng, model, spl, state)
            δacc .+= t.δ; βacc .+= t.β; cnt += 1
            append!(τ²_excluded, t.τ²[t.δ .== 0])
        end
        @test length(t.τ²) == p
        @test all(>(0), t.τ²)
        @test !hasproperty(t, :σ²ᵦ)
        @test all(t.β[t.δ .== 0] .== 0)
        top = sortperm(δacc ./ cnt; rev=true)[1:8]
        @test length(intersect(top, true_idx)) >= 6
        @test cor(βacc ./ cnt, β_true) > 0.6
        # An excluded marker's τ² is drawn from its prior.
        prior = InverseGamma(3.0 / 2, 3.0 * S2b / 2)
        @test median(τ²_excluded) ≈ median(prior) rtol = 0.05
    end

    @testset "BayesB and BayesCπ are different samplers" begin
        rng0 = StableRNG(5)
        n, p = 80, 30
        X = randn(rng0, n, p); X .-= mean(X; dims=1)
        y = X[:, 1] .* 2 .+ X[:, 2] .* 0.3 .+ randn(rng0, n)
        mB, sB = bayes_alphabet_model(y, X; variant=:B, S2b=0.5)
        mC, sC = bayes_alphabet_model(y, X; variant=:C, S2b=0.5)
        function run(m, s)
            rng = StableRNG(1)
            t, st = AbstractMCMC.step(rng, m, s)
            for _ in 1:50; t, st = AbstractMCMC.step(rng, m, s, st); end
            t
        end
        tB, tC = run(mB, sB), run(mC, sC)
        @test haskey(tB, :τ²) && !haskey(tB, :σ²ᵦ)
        @test haskey(tC, :σ²ᵦ) && !haskey(tC, :τ²)
        @test tB.β != tC.β
        @test_throws ArgumentError bayes_alphabet_model(y, X; variant=:A)
    end

    @testset "genomic_relatedness on a VanRaden G with default arguments" begin
        for (n, m) in ((30, 50), (60, 2000))
            rng = StableRNG(n)
            M = Float64.(rand(rng, 0:2, n, m))
            pfreq = vec(mean(M; dims=1)) ./ 2
            Zc = M .- 2 .* pfreq'
            G = Zc * Zc' ./ (2 * sum(pfreq .* (1 .- pfreq)))
            rel = genomic_relatedness(G)
            Gb = 0.99 .* G + 0.01I
            @test rel.kind == :genomic
            @test rel.Kinv * Gb ≈ I atol = 1e-6
            # The markers entry point builds the same G.
            @test genomic_relatedness_from_markers(M).Kinv ≈ rel.Kinv rtol = 1e-8
        end
        @test_throws ArgumentError genomic_relatedness([1.0 2.0; 2.0 1.0]; blend=0.0)
        @test_throws ArgumentError genomic_relatedness([1.0 0.0; 0.0 1.0]; blend=1.5)
    end

    @testset "singlestep_relatedness matches the dense H⁻¹ and H" begin
        rng = StableRNG(13)
        q = 40
        sire = zeros(Int, q); dam = zeros(Int, q)
        for i in 5:q
            sire[i] = rand(rng, 1:i-1); dam[i] = rand(rng, 1:i-1)
            while dam[i] == sire[i]; dam[i] = rand(rng, 1:i-1); end
        end
        A = dense_A(sire, dam)
        g = sort(randperm(rng, q)[1:15])
        M = Float64.(rand(rng, 0:2, length(g), 300))
        pfreq = vec(mean(M; dims=1)) ./ 2
        Zc = M .- 2 .* pfreq'
        G = Zc * Zc' ./ (2 * sum(pfreq .* (1 .- pfreq)))

        blend = 0.05
        rel = singlestep_relatedness(sire, dam, G, g; blend=blend)
        @test rel.kind == :singlestep
        A22 = A[g, g]
        Gb = (1 - blend) .* G .+ blend .* A22
        Hinv_ref = inv(A)
        Hinv_ref[g, g] .+= inv(Gb) .- inv(A22)
        @test rel.Kinv ≈ Hinv_ref rtol = 1e-7
        # Legarra et al. (2009): H = A + A₍·₂₎ A₂₂⁻¹ (G − A₂₂) A₂₂⁻¹ A₍₂·₎
        Q = A[:, g] / A22
        H = A .+ Q * (Gb .- A22) * Q'
        @test inv(Symmetric(rel.Kinv)) ≈ H rtol = 1e-6
    end
end
