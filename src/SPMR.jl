module SPMR

using LinearAlgebra, SparseArrays
using LinearAlgebra: checksquare

const FloatMatrix = AbstractMatrix{Float64}
const RealMatrix = AbstractMatrix{<:Real}
const FloatFact = Factorization{Float64}
const RealFact = Factorization{<:Real}

@enum SPMRFlag CONVERGED=0 MAXIT_EXCEEDED=1 OTHER=2

struct SPMRResult
    x::Vector{Float64}
    y::Vector{Float64}
    flag::SPMRFlag
    iter::Int
    resvec::Vector{Float64}
end

include("saddlepoint.jl")

include("simba.jl")
include("spmr.jl")

end # module
