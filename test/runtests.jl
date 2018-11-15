using Test, LinearAlgebra, SPMR

@testset "SPMatrix construction" begin
    A = SymTridiagonal(fill(2, 3), fill(1, 2))
    G₁ᵀ = Bidiagonal(fill(4, 3), fill(3, 2), :U)
    G₂ = Bidiagonal(fill(2, 3), fill(1, 2), :L)

    K = SPMatrix(A, G₁ᵀ, G₂)

    @test size(K) == (6, 6)
end
