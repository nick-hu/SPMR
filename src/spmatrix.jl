# Saddle-point matrices

export SPMatrix

struct SPMatrix{T<:FloatInvOperator, U<:FloatInvOperator, V<:FloatOperator, W<:FloatOperator}
    A::T
    Aᵀ::U
    G₁ᵀ::V
    G₂::W

    function SPMatrix{T, U, V, W}(A, Aᵀ, G₁ᵀ, G₂) where {T<:FloatInvOperator,
                                                         U<:FloatInvOperator,
                                                         V<:FloatOperator,
                                                         W<:FloatOperator}
        n = checksquare(A)
        m = size(G₂, 1)

        if size(G₁ᵀ) ≠ (n, m)
            throw(DimensionMismatch("G₁ᵀ should have size ($n, $m)"))
        elseif size(G₂) ≠ (m, n)
            throw(DimensionMismatch("G₂ should have size ($m, $n)"))
        end

        if !checktranspose(G₁ᵀ)
            error("Left-multiplication by G₁ should be implemented")
        elseif !checktranspose(G₂)
            error("Left-multiplication by G₂ᵀ should be implemented")
        end

        return new{T, U, V, W}(A, Aᵀ, G₁ᵀ, G₂)
    end
end

function SPMatrix(A::T, Aᵀ::U, G₁ᵀ::V, G₂::W) where {T<:FloatInvOperator,
                                                     U<:FloatInvOperator,
                                                     V<:FloatOperator,
                                                     W<:FloatOperator}
    return SPMatrix{T, U, V, W}(A, Aᵀ, G₁ᵀ, G₂)
end

function SPMatrix(A::T, G₁ᵀ::V, G₂::W) where {T<:InvLinearMap{Float64},
                                              V<:RealOperator,
                                              W<:RealOperator}
    if !checktranspose(A.map)
        error("Left-division by Aᵀ should be implemented")
    end

    Aᵀ = issymmetric(A.map) ? A : InvLinearMap(transpose(A.map))

    return SPMatrix(A, Aᵀ, convert(FloatOperator, G₁ᵀ), convert(FloatOperator, G₂))
end

function SPMatrix(A::T, G₁ᵀ::V, G₂::W) where {T<:FloatMatrix,
                                              V<:FloatOperator,
                                              W<:FloatOperator}
    F = factorize(A)
    Fᵀ = issymmetric(A) ? F : factorize(copy(transpose(A)))

    return SPMatrix(F, Fᵀ, G₁ᵀ, G₂)
end

function SPMatrix(A::T, G₁ᵀ::V, G₂::W) where {T<:RealMatrix,
                                              V<:RealOperator,
                                              W<:RealOperator}
    return SPMatrix(convert(FloatMatrix, A),
                    convert(FloatOperator, G₁ᵀ),
                    convert(FloatOperator, G₂))
end

function SPMatrix(K::RealMatrix, n::Integer)
    return SPMatrix(K[1:n, 1:n], K[1:n, n+1:end], K[n+1:end, 1:n])
end

SPMatrix(K::SPMatrix) = K

Base.size(K::SPMatrix) = (size(K.A, 1) + size(K.G₂, 1),
                          size(K.A, 2) + size(K.G₁ᵀ, 2))

block_sizes(K::SPMatrix) = (size(K.A, 1), size(K.G₂, 1))

function Base.Matrix(K::SPMatrix)
    _, m = block_sizes(K)

    return [convert(Matrix, K.A) convert(Matrix, K.G₁ᵀ);
            convert(Matrix, K.G₂) zeros(m, m)]
end

function SparseArrays.sparse(K::SPMatrix)
    _, m = block_sizes(K)

    try
        return [sparse(K.A) sparse(K.G₁ᵀ);
                sparse(K.G₂) spzeros(m, m)]
    catch
        return [sparse(convert(Matrix, K.A)) sparse(K.G₁ᵀ);
                sparse(K.G₂) spzeros(m, m)]
    end
end

Base.Array(K::SPMatrix) = Matrix(K)
Base.convert(::Type{Matrix}, K::SPMatrix) = Matrix(K)
Base.convert(::Type{Array}, K::SPMatrix) = Array(K)
Base.convert(::Type{SparseMatrixCSC}, K::SPMatrix) = sparse(K)

function checktranspose(A::RealOperator)
    return !(isa(A, FunctionMap) && !issymmetric(A) && A.fc == nothing)
end
