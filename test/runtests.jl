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

        @test K == SPMatrix(K)
    end

    @testset "Construction from rectangular matrix" begin
        @test_throws DimensionMismatch K = SPMatrix(rand(4, 3), 2)
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

    @testset "Set in zero block" begin
        K = SPMatrix(zeros(4, 4), 1)

        @test_throws ArgumentError K[2, 2] = 1
    end

end
