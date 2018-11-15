using Test, SPMR, LinearAlgebra

@testset "SPMatrix construction" begin
    A = SymTridiagonal(fill(2.0, 3), fill(1.0, 2))
    G₁ᵀ = Bidiagonal(fill(4.0, 3), fill(3.0, 2), :U)
    G₂ = Bidiagonal(fill(2.0, 3), fill(1.0, 2), :L)

    K = SPMatrix{typeof(A), typeof(G₁ᵀ), typeof(G₂)}(A, G₁ᵀ, G₂)

    @test size(K) == (6, 6)

    println(K)
end
