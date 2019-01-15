# Grcar example using -NS family methods

using LinearAlgebra, SparseArrays

using SPMR, LinearMaps

include("grcar.jl")

n, m = 2000, 1000
#n, m = 200, 100

A = grcar(n)

#F = sprand(m, n÷2, 0.1) + 100I
F = Matrix(100I, m, m)
G₁ = [F F]

function kernel_project(G::Matrix{<:Real})
    m, n = size(G)

    M = [I G'; G zeros(m, m)]
    F = factorize(M)
    d(c::Vector{Float64}) = (F \ [c; zeros(m)])[1:n]

    return d
end

#@time H₁ = nullspace(G₁)

#A_map = LinearMap(x -> A * x, x -> A' * x, n)
#H₁_map = LinearMap(x -> H₁ * x, x -> H₁' * x, n, m)

#K = SpmrNsMatrix(A, H₁, H₁, m)
#K = SpmrNsMatrix(A_map, H₁_map, H₁_map, m)

K = SpmrNsMatrix(A,
                 LinearMap(kernel_project(G₁), n, n, issymmetric=true),
                 LinearMap(kernel_project(G₁), n, n, issymmetric=true),
                 m)

#f = Vector(0.005:0.005:1.0)
f = Vector(0.0005:0.0005:1.0)
#f = rand(n)

b = [f; zeros(m)]

@time result = spmr_ns(K, f, tol=1e-10, maxit=10)
