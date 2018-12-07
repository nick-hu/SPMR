__precompile__()

module SPMR

using LinearAlgebra, SparseArrays
using LinearAlgebra: checksquare

using LinearMaps: LinearMap, FunctionMap, A_mul_B!, At_mul_B!, Ac_mul_B!

include("invfunctionmap.jl")


const FloatMatrix = AbstractMatrix{Float64}
const RealMatrix = AbstractMatrix{<:Real}

const FloatOperator = Union{FloatMatrix, LinearMap{Float64}}
const RealOperator = Union{RealMatrix, LinearMap{<:Real}}
const FloatInvOperator = Union{Factorization{Float64}, InvLinearMap{Float64}}
const RealInvOperator = Union{Factorization{<:Real}, InvLinearMap{<:Real}}

function Base.convert(::Type{FloatOperator}, A::RealOperator)
    isa(A, RealMatrix) ? convert(FloatMatrix, A) : convert(LinearMap{Float64}, A)
end


@enum SPMRFlag begin
    CONVERGED
    MAXIT_EXCEEDED
    OTHER
end

struct SPMRResult
    x::Vector{Float64}
    y::Vector{Float64}
    flag::SPMRFlag
    iter::Int
    resvec::Vector{Float64}
end

include("spmatrix.jl")

include("simba.jl")
include("spmr.jl")

end # module
