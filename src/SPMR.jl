__precompile__()

module SPMR

using LinearAlgebra, SparseArrays
using LinearMaps: LinearMap

include("invfunctionmap.jl")

const FloatOperator = Union{AbstractMatrix{Float64}, LinearMap{Float64}}
const RealOperator = Union{AbstractMatrix{<:Real}, LinearMap{<:Real}}
const FloatInvOperator = Union{Factorization{Float64}, InvLinearMap{Float64}}
const RealInvOperator = Union{Factorization{<:Real}, InvLinearMap{<:Real}}

function Base.convert(::Type{FloatOperator}, A::RealOperator)
    if isa(A, AbstractMatrix{<:Real})
        return convert(AbstractMatrix{Float64}, A)
    else
        return convert(LinearMap{Float64}, A)
    end
end

include("spmatrix.jl")

const type_map = (:simba_sc => (:SpmrScMatrix, :SimbaScIterator, :SpmrScIterate),
                  :simba_ns => (:SpmrNsMatrix, :SimbaNsIterator, :SpmrNsIterate),
                  :simbo_sc => (:SpmrScMatrix, :SimboScIterator, :SpmrScIterate),
                  :simbo_ns => (:SpmrNsMatrix, :SimboNsIterator, :SpmrNsIterate))

include("simbidiag.jl")
include("spmr.jl")

end # module
