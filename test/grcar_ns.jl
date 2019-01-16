# Grcar example using -NS family methods

using LinearAlgebra, SparseArrays, Profile, ProfileView

using SPMR, LinearMaps

include("null_proj.jl")
include("grcar.jl")

n, m = 2000, 1000
#n, m = 200, 100

A = grcar(n)

F = sparse(100I, m, m)
#F = sprand(m, n÷2, 0.1) + 100I
G₁ = [F F]

@time H₁ = nullspace(Matrix(G₁))
K = SpmrNsMatrix(A, H₁, H₁, m)

#=
@time H₁ = nullspace(G₁)
A_map = LinearMap(x -> A * x, x -> A' * x, n)
H₁_map = LinearMap(x -> H₁ * x, x -> H₁' * x, n, m)
K = SpmrNsMatrix(A_map, H₁_map, H₁_map, m)
=#

#=
K = SpmrNsMatrix(A,
                 LinearMap(null_proj(G₁), n, n, issymmetric=true),
                 LinearMap(null_proj(G₁), n, n, issymmetric=true),
                 m)
=#

f = Vector(0.0005:0.0005:1.0)
#f = Vector(0.005:0.005:1.0)
#f = rand(n)

@time result = spmr_ns(K, f, tol=1e-10, maxit=10)

#=
Profile.clear()
@profile result = spmr_ns(K, f, tol=1e-10, maxit=10)
ProfileView.view();
=#
