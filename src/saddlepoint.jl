# Saddle-point matrices

struct SPMatrix{T<:AbstractMatrix{Float64},
                U<:AbstractMatrix{Float64},
                V<:AbstractMatrix{Float64}} <: AbstractMatrix{Float64}
    A::T
    G₁ᵀ::U
    G₂::V

    n::Int
    m::Int

    function SPMatrix{T, U, V}(A, G₁ᵀ, G₂) where {T<:AbstractMatrix{Float64},
                                                  U<:AbstractMatrix{Float64},
                                                  V<:AbstractMatrix{Float64}}

        n = checksquare(A)
        m = size(G₁ᵀ, 2)

        @assert size(G₁ᵀ) == (n, m)
        @assert size(G₂) == (m, n)

        new{T, U, V}(A, G₁ᵀ, G₂, size(A, 1), size(G₁ᵀ, 2))
    end
end

Base.size(K::SPMatrix) = (K.n + K.m, K.n + K.m)

function Base.getindex(K::SPMatrix, i::Integer, j::Integer)
    if !(1 ≤ i ≤ size(K, 1) && 1 ≤ j ≤ size(K, 2))
        throw(BoundsError(K, (i, j)))
    end

    if i ≤ K.n && j ≤ K.n
        return K.A[i, j]
    elseif i > K.n && j ≤ K.n
        return K.G₂[i-K.n, j]
    elseif i ≤ K.n && j > K.n
        return K.G₁ᵀ[i, j-K.n]
    else
        return zero(Float64)
    end
end

function Base.setindex!(K::SPMatrix, x::Real, i::Integer, j::Integer)
    @boundscheck checkbounds(K, i, j)

    if i ≤ K.n && j ≤ K.n
        @inbounds K.A[i, j] = x
    elseif i > K.n && j ≤ K.n
        @inbounds K.G₂[i-K.n, j] = x
    elseif i ≤ K.n && j > K.n
        @inbounds K.G₁ᵀ[i, j-K.n] = x
    elseif !iszero(x)
        throw(ArgumentError(string("cannot set entry ($i, $j) in zero block",
                                   "to a nonzero value ($x)")))
    end

    return x
end
