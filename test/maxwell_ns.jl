# Maxwell example using -NS family methods

using LinearAlgebra, SparseArrays, JLD

using SPMR, LinearMaps

@load "test/matrices_homogeneous3.jld"

G₁ = B

function kernel_project(G::AbstractMatrix{<:Real})
    m, n = size(G)

    M = [I G'; G zeros(m, m)]
    F = factorize(M)
    d(c::Vector{Float64}) = (F \ [c; zeros(m)])[1:n]

    return d
end

@time H₁ = nullspace(Matrix(G₁))

#A_map = LinearMap(x -> A * x, x -> A' * x, n)
#H₁_map = LinearMap(x -> H₁ * x, x -> H₁' * x, n, m)

K = SpmrNsMatrix(A, H₁, H₁, m)
#K = SpmrNsMatrix(A_map, H₁_map, H₁_map, m)

#=
K = SpmrNsMatrix(A,
                 LinearMap(kernel_project(G₁), n, n, issymmetric=true),
                 LinearMap(kernel_project(G₁), n, n, issymmetric=true),
                 m)
=#

f = Vector(range(-1, 1, length=n))

b = [f; zeros(m)]

@time result = spmr_ns(K, f, tol=1e-10, maxit=20)
