# Grcar example using -NS family methods

using LinearAlgebra, SparseArrays, Profile, ProfileView

using SPMR, LinearMaps

include("null_proj.jl")
include("grcar.jl")

#n, m = 2000, 1000
n, m = 1000, 500  # Benchmark sizes

A = grcar(n)

F = sparse(100I, m, m)
#F = sprand(m, n÷2, 0.1) + 100I
G₁ = [F zeros(m, m)]

#=
@time H₁ = nullspace(Matrix(G₁))
K = SpmrNsMatrix(A, H₁, H₁, m)
=#

#=
@time H₁ = nullspace(Matrix(G₁))
A_map = LinearMap(x -> A * x, x -> A' * x, n)
H₁_map = LinearMap(x -> H₁ * x, x -> H₁' * x, n, m)
K = SpmrNsMatrix(A_map, H₁_map, H₁_map, m)
=#

H₁ = LinearMap(null_proj_cp(G₁), n, issymmetric=true)
K = SpmrNsMatrix(A, H₁, H₁, m)

f = Vector(range(-1, 1, length=n))
#f = rand(n)

@time result = spmr_ns(K, f, tol=1e-10, maxit=2m)

#=
Profile.clear()
@profile result = spmr_ns(K, f, tol=1e-10, maxit=10)
ProfileView.view();
=#
