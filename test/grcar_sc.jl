# Grcar example using -SC family methods

using LinearAlgebra, SparseArrays, Profile, ProfileView

using SPMR, LinearMaps

include("grcar.jl")

#n, m = 2000, 1000
#n, m = 200, 100
n, m = 1000, 500

A = grcar(n)

F = sparse(100I, m, m)
#F = sprand(m, n÷2, 0.1) + 100I
G₁ = [F F]

K = SpmrScMatrix(A, G₁', G₁)

#=
A⁻¹_map = InvLinearMap(b -> A \ b, b -> A' \ b, n)
G₁ᵀ_map = LinearMap(x -> G₁' * x, x -> G₁ * x, n, m)
G₂_map = LinearMap(x -> G₁ * x, x -> G₁' * x, m, n)
K = SpmrScMatrix(A⁻¹_map, G₁ᵀ_map, G₂_map)
=#

g = ones(m)
#g = rand(m)

@time result = spmr_sc(K, g, tol=1e-10, maxit=2m)

#=
Profile.clear()
@profile result = spmr_sc(K, g, tol=1e-10, maxit=2m)
ProfileView.view()
=#
