# Saddle-point matrices

struct SPMatrix{T<:AbstractMatrix{Float64},
                U<:AbstractMatrix{Float64},
                V<:AbstractMatrix{Float64}} <: AbstractMatrix{Float64}
    A::T
    G₁ᵀ::U
    G₂::V

    function SPMatrix{T, U, V}(A, G₁ᵀ, G₂) where {T<:AbstractMatrix{Float64},
                                                  U<:AbstractMatrix{Float64},
                                                  V<:AbstractMatrix{Float64}}
        n = checksquare(A)
        m = size(G₁ᵀ, 2)

        @assert size(G₁ᵀ) == (n, m)
        @assert size(G₂) == (m, n)

        return new{T, U, V}(A, G₁ᵀ, G₂)
    end
end

function SPMatrix(A::T, G₁ᵀ::U, G₂::V) where {T<:AbstractMatrix{Float64},
                                              U<:AbstractMatrix{Float64},
                                              V<:AbstractMatrix{Float64}}
    return SPMatrix{T, U, V}(A, G₁ᵀ, G₂)
end

function SPMatrix(A::T, G₁ᵀ::U, G₂::V) where {T<:AbstractMatrix{<:Real},
                                              U<:AbstractMatrix{<:Real},
                                              V<:AbstractMatrix{<:Real}}
    return SPMatrix(map(x -> convert(AbstractMatrix{Float64}, x), (A, G₁ᵀ, G₂))...)
end

function SPMatrix(K::AbstractMatrix{<:Real}, n::Integer)
    return SPMatrix(K[1:n, 1:n], K[1:n, n+1:end], K[n+1:end, 1:n])
end

SPMatrix(K::SPMatrix) = K

Base.size(K::SPMatrix) = (size(K.A, 1) + size(K.G₂, 1),
                          size(K.A, 2) + size(K.G₁ᵀ, 2))

function Base.getindex(K::SPMatrix, i::Integer, j::Integer)
    if !(1 ≤ i ≤ size(K, 1) && 1 ≤ j ≤ size(K, 2))
        throw(BoundsError(K, (i, j)))
    end

    n = size(K.A, 1)

    if i ≤ n && j ≤ n
        return K.A[i, j]
    elseif i > n && j ≤ n
        return K.G₂[i-n, j]
    elseif i ≤ n && j > n
        return K.G₁ᵀ[i, j-n]
    else
        return zero(Float64)
    end
end

function Base.setindex!(K::SPMatrix, x, i::Integer, j::Integer)
    @boundscheck checkbounds(K, i, j)

    n = size(K.A, 1)

    if i ≤ n && j ≤ n
        @inbounds K.A[i, j] = x
    elseif i > n && j ≤ n
        @inbounds K.G₂[i-n, j] = x
    elseif i ≤ n && j > n
        @inbounds K.G₁ᵀ[i, j-n] = x
    elseif !iszero(x)
        throw(ArgumentError(string("cannot set entry ($i, $j) in zero block ",
                                   "to a nonzero value ($x)")))
    end

    return x
end

function Base.replace_in_print_matrix(K::SPMatrix, i::Integer, j::Integer,
                                      s::AbstractString)
    n = size(K.A, 1)
    return i > n && j > n ? Base.replace_with_centered_mark(s) : s
end
