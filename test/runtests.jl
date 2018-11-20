using Test, LinearAlgebra, SPMR

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

@testset "SPMatrix get/set" begin

    @testset "Basic get/set" begin
        K = SPMatrix(zeros(4, 4), 3)

        K[1, 1] = 1
        K[1, 4] = 2
        K[4, 1] = 3

        @test K.A[1, 1] == 1
        @test K.G₁ᵀ[1, 1] == 2
        @test K.G₂[1, 1] == 3
        @test K[4, 4] == 0
    end

    @testset "Out of bounds get/set" begin
        K = SPMatrix(zeros(4, 4), 3)

        @test_throws BoundsError K[5, 4]
        @test_throws BoundsError K[4, 5]
        @test_throws BoundsError K[5, 4] = 0
        @test_throws BoundsError K[4, 5] = 0
    end

    @testset "Set in zero block" begin
        K = SPMatrix(zeros(4, 4), 1)

        @test_throws ArgumentError K[2, 2] = 1
    end

end

@testset "SIMBA-SC" begin
    n, m = 3, 2
    K = SPMatrix(rand(n+m, n+m), n)
    A, G₁ᵀ, G₂ = K.A, K.G₁ᵀ, K.G₂
    b, c = rand(m), rand(m)

    @testset "Iterator construction" begin
        SSI = simba_sc(K, b, c)

        @test Matrix(SSI.A) ≈ A
        @test SSI.G₁ᵀ == G₁ᵀ
        @test SSI.G₂ == G₂
    end

    SSI = simba_sc(K, b, c)

    @testset "Iteration" begin
        k = 0

        for _ in SSI
            k += 1
        end

        @test length(SSI) == k
    end

    @testset "Iteration correctness" begin
        SI_prev = eltype(typeof(SSI))()

        for (k, SI) in enumerate(SSI)
            @test SI.k == k

            if k == 1
                @test SI.β * SI.v ≈ b
                @test SI.δ * SI.z ≈ c

                @test A * (copysign(SI.α, SI.ξ) * SI.u) ≈ G₁ᵀ * SI.v
                @test A' * (copysign(SI.γ, SI.ξ) * SI.w) ≈ G₂' * SI.z
            else
                @test SI.β * SI.v ≈ G₁ᵀ' * SI_prev.w - SI_prev.α * SI_prev.v
                @test SI.δ * SI.z ≈ G₂ * SI_prev.u - SI_prev.γ * SI_prev.z

                @test A * (copysign(SI.α, SI.ξ) * SI.u +
                           copysign(SI.β, SI_prev.ξ) * SI_prev.u) ≈ G₁ᵀ * SI.v
                @test A' * (copysign(SI.γ, SI.ξ) * SI.w +
                            copysign(SI.δ, SI_prev.ξ) * SI_prev.w) ≈ G₂' * SI.z
            end

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
