# Maxwell example using -NS family methods

using LinearAlgebra, SparseArrays, Profile, ProfileView, JLD

using SPMR, LinearMaps

include("null_proj.jl")
@load "test/matrices_homogeneous3.jld"  # A, B, f, m, n

G₁ = B

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

K = SpmrNsMatrix(A,
                 LinearMap(null_proj(G₁), n, n, issymmetric=true),
                 LinearMap(null_proj(G₁), n, n, issymmetric=true),
                 m)

f = Vector(range(-1, 1, length=n))

@time result = spmr_ns(K, f, tol=1e-10, maxit=20)

#=
Profile.clear()
@profile result = spmr_ns(K, f, tol=1e-10, maxit=10)
ProfileView.view();
=#
