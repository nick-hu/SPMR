using LinearMaps: FunctionMap, A_mul_B!

export InvLinearMap

"""
Type representing the inverse of a linear map ``A: \\mathbb{R}^M \\to \\mathbb{R}^M``.
"""
struct InvLinearMap{T}
    map::LinearMap{T}
end

"""
    InvLinearMap(f::Function, M::Int; <keyword arguments>)

Construct an [`InvLinearMap`](@ref) with ``A^{-1} = f`` and ``M`` as above.

The keyword arguments are as for a `LinearMaps.LinearMap`.
"""
function InvLinearMap(f::Function, M::Int; kwargs...)
    return InvLinearMap(FunctionMap{Float64}(f, M; kwargs...))
end

"""
    InvLinearMap(f::Function, fc::Function, M::Int; <keyword arguments>)

Construct an [`InvLinearMap`](@ref) with ``A^{-1} = f``, ``A^{-T} = fc``,
and ``M`` as above.

The keyword arguments are as for a `LinearMaps.LinearMap`.
"""
function InvLinearMap(f::Function, fc::Function, M::Int; kwargs...)
    return InvLinearMap(FunctionMap{Float64}(f, fc, M; kwargs...))
end

Base.size(A::InvLinearMap, args...) = size(A.map, args...)

function LinearAlgebra.ldiv!(y::AbstractVector, A::InvLinearMap, x::AbstractVector)
    return A_mul_B!(y, A.map, x)
end

function LinearAlgebra.ldiv!(A::InvLinearMap, x::AbstractVector)
    return A_mul_B!(x, A.map, x)
end
