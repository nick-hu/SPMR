using LinearMaps: FunctionMap, A_mul_B!

export InvLinearMap

struct InvLinearMap{T}
    map::LinearMap{T}
end

function InvLinearMap(f::Function, M::Int; kwargs...)
    return InvLinearMap(FunctionMap{Float64}(f, nothing, M; kwargs...))
end

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
