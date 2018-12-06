# Grcar benchmark

function grcar(n::Int, k::Int=3)
    A = spzeros(Int, n, n)

    @inbounds A[diagind(A, -1)] .= -1

    @inbounds for i in 0:k
        A[diagind(A, i)] .= 1
    end

    return A
end

n, m = 2000, 1000

A = grcar(n)
#F = sprand(m, n÷2, 0.1) + 100I
F = Matrix(100I, m, m)
G₁ = [F F]
K = SPMatrix(A, G₁', G₁)

#g = rand(m)
g = ones(m)

b = [zeros(n); g]
x_exact = K \ b

#=
Profile.clear()
@profile result = spmr_sc(K, g, tol=1e-10, maxit=2m)
ProfileView.view()
=#
@time result = spmr_sc(K, g, tol=1e-10, maxit=2m)
#println(result.resvec)
