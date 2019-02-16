__precompile__()

module SPMR

using LinearAlgebra, SparseArrays
using LinearMaps: LinearMap

include("invfunctionmap.jl")

const FloatOperator = Union{AbstractMatrix{Float64}, LinearMap{Float64}}
const FloatInvOperator = Union{Factorization{Float64}, InvLinearMap{Float64},
                               UniformScaling}

"""An `AbstractMatrix{<:Real}` or a `LinearMaps.LinearMap{<:Real}`."""
const RealOperator = Union{AbstractMatrix{<:Real}, LinearMap{<:Real}}

"""A `Factorization{<:Real}`, `SPMR.InvLinearMap{<:Real}`, or `UniformScaling`."""
const RealInvOperator = Union{Factorization{<:Real}, InvLinearMap{<:Real},
                              UniformScaling}

function Base.convert(::Type{FloatOperator}, A::RealOperator)
    if isa(A, AbstractMatrix{<:Real})
        return convert(AbstractMatrix{Float64}, A)
    else
        return convert(LinearMap{Float64}, A)
    end
end

# ldiv! for UniformScaling matrices (e.g., when no preconditioner is specified)

function LinearAlgebra.ldiv!(y::AbstractVector, A::UniformScaling, x::AbstractVector)
    y .= x / A.λ
    return y
end

include("spmatrix.jl")
include("simbidiag.jl")
include("spmr.jl")

end # module
