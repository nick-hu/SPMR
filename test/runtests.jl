using Test, LinearAlgebra, SparseArrays

using SPMR
using SPMR: simba_sc

@testset "SPMatrix construction" begin

    @testset "Construction from float matrix" begin
        M = rand(2, 2)
        K = SPMatrix(M, M, M)

        @test size(K) == (4, 4)
    end

    @testset "Construction from real matrix" begin
        M = rand(Int, 2, 2)
        K = SPMatrix(M, M, M)

        @test size(K) == (4, 4)
    end

    @testset "Conversion from square matrix" begin
        K = SPMatrix(rand(4, 4), 3)

        @test size(K) == (4, 4)
        @test size(K.A) == (3, 3)
        @test size(K.G₁ᵀ) == (3, 1)
        @test size(K.G₂) == (1, 3)
    end

    @testset "Conversion from self" begin
        K = SPMatrix(rand(4, 4), 2)

        @test SPMatrix(K) == K
    end

    @testset "Construction from rectangular matrix" begin
        @test_throws DimensionMismatch K = SPMatrix(rand(4, 3), 2)
        @test_throws DimensionMismatch K = SPMatrix(rand(2, 2), rand(1, 2), rand(2, 2))
    end

end

@testset "SIMBA-SC" begin
    n, m = 3, 2
    K = SPMatrix(rand(n+m, n+m), n)
    A, G₁ᵀ, G₂ = Matrix(K.A), K.G₁ᵀ, K.G₂
    b, c = rand(m), rand(m)

    @testset "Iterator construction" begin
        SSI, _ = simba_sc(K, b, c)

        @test Matrix(SSI.K.A) == A
        @test SSI.K.G₁ᵀ == G₁ᵀ
        @test SSI.K.G₂ == G₂
    end

    SSI, SI₀ = simba_sc(K, b, c)

    @testset "Iteration" begin
        k = 0

        for _ in SSI
            k += 1
        end

        @test length(SSI) == k
    end

    @testset "Iteration correctness" begin
        SI_prev = SI₀

        @test SI_prev.β * SI_prev.v ≈ b
        @test SI_prev.δ * SI_prev.z ≈ c

        @test A * (copysign(SI_prev.α, SI_prev.ξ) * SI_prev.u) ≈ G₁ᵀ * SI_prev.v
        @test A' * (copysign(SI_prev.γ, SI_prev.ξ) * SI_prev.w) ≈ G₂' * SI_prev.z

        for (k, SI) in enumerate(SSI)
            @test SI.β * SI.v ≈ G₁ᵀ' * SI_prev.w - SI_prev.α * SI_prev.v
            @test SI.δ * SI.z ≈ G₂ * SI_prev.u - SI_prev.γ * SI_prev.z

            @test A * (copysign(SI.α, SI.ξ) * SI.u +
                       copysign(SI.β, SI_prev.ξ) * SI_prev.u) ≈ G₁ᵀ * SI.v
            @test A' * (copysign(SI.γ, SI.ξ) * SI.w +
                        copysign(SI.δ, SI_prev.ξ) * SI_prev.w) ≈ G₂' * SI.z

            SI_prev = SI
        end
    end

    @testset "Almost-conjugacy and orthonormality" begin
        for SI in SSI
            @test abs(SI.w ⋅ (A * SI.u)) ≈ 1
            @test norm(SI.v) ≈ 1
            @test norm(SI.z) ≈ 1
        end
    end
end

@testset "SPMR-SC" begin
    n, m = 100, 70
    K = SPMatrix(rand(n+m, n+m), n)
    g = rand(m)

    @testset "Relative residual" begin
        result = spmr_sc(K, g)

        b = [zeros(n); g]
        x = [result.x; result.y]

        @test norm(b - Matrix(K) * x) / norm(b) ≈ result.resvec[end]
    end

    @testset "Monotonic decrease of residual norm" begin
        result = spmr_sc(K, g)

        resvec_next = circshift(result.resvec, (-1))
        resvec_next[end] = zero(Float64)

        @test all(result.resvec .≥ resvec_next)
    end
end
