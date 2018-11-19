# Saddle-point matrices

struct SPMatrix{T<:FloatMatrix, U<:FloatMatrix, V<:FloatMatrix} <: FloatMatrix
    A::T
    G₁ᵀ::U
    G₂::V

    function SPMatrix{T, U, V}(A, G₁ᵀ, G₂) where {T<:FloatMatrix, U<:FloatMatrix,
                                                  V<:FloatMatrix}
        n = checksquare(A)
        m = size(G₁ᵀ, 2)

        if size(G₁ᵀ) ≠ (n, m)
            throw(DimensionMismatch("G₁ᵀ should have size ($n, $m)"))
        elseif size(G₂) ≠ (m, n)
            throw(DimensionMismatch("G₂ should have size ($m, $n)"))
        end

        return new{T, U, V}(A, G₁ᵀ, G₂)
    end
end

function SPMatrix(A::T, G₁ᵀ::U, G₂::V) where {T<:FloatMatrix, U<:FloatMatrix,
                                              V<:FloatMatrix}
    return SPMatrix{T, U, V}(A, G₁ᵀ, G₂)
end

function SPMatrix(A::T, G₁ᵀ::U, G₂::V) where {T<:RealMatrix, U<:RealMatrix, V<:RealMatrix}
    return SPMatrix(map(x -> convert(FloatMatrix, x), (A, G₁ᵀ, G₂))...)
end

function SPMatrix(K::RealMatrix, n::Integer)
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
