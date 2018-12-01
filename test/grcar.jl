# Grcar matrix tests

function grcar(n::Int, k::Int=3)
    A = spzeros(Int, n, n)

    @inbounds A[diagind(A, -1)] .= -1

    @inbounds for i in 0:k
        A[diagind(A, i)] .= 1
    end

    return A
end

@testset "Grcar: SPMR-SC" begin
    n, m = 1000, 500

    A = grcar(n)
    F = sprand(m, n÷2, 0.1) + 100I
    G₁ = [F F]
    K = SPMatrix(A, G₁', G₁)

    g = rand(m)

    x_exact = K \ [zeros(n); g]

    result = spmr_sc(K, g, tol=1e-10, maxit=2m)
end
