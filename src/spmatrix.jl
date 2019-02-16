# Saddle-point matrices

using LinearAlgebra: checksquare

export
    SpmrMatrix,
    SpmrScMatrix, SpmrNsMatrix

abstract type SpmrMatrix end

# -SC family

"""
Saddle-point matrix type for SPMR-SC and SPQMR-SC representing a block matrix of the form
```math
\\begin{bmatrix}
A & G_1^T \\\\ G_2 & 0
\\end{bmatrix},
```
where ``A \\in \\mathbb{R}^{n \\times n}`` and ``G_1, G_2 \\in \\mathbb{R}^{m \\times n}``.
"""
struct SpmrScMatrix{T<:FloatInvOperator,
                    U<:FloatInvOperator,
                    V<:FloatOperator,
                    W<:FloatOperator} <: SpmrMatrix
    A::T
    Aᵀ::U
    G₁ᵀ::V
    G₂::W

    function SpmrScMatrix{T, U, V, W}(A, Aᵀ, G₁ᵀ, G₂) where {T<:FloatInvOperator,
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

function SpmrScMatrix(A::T, Aᵀ::U, G₁ᵀ::V, G₂::W) where {T<:FloatInvOperator,
                                                         U<:FloatInvOperator,
                                                         V<:FloatOperator,
                                                         W<:FloatOperator}
    return SpmrScMatrix{T, U, V, W}(A, Aᵀ, G₁ᵀ, G₂)
end

"""
    SpmrScMatrix(A::InvLinearMap{Float64}, G₁ᵀ::RealOperator, G₂::RealOperator)

Construct an [`SpmrScMatrix`](@ref) from an [`InvLinearMap`](@ref) representing ``A^{-1}`` and
[`RealOperator`](@ref)s representing ``G_1^T`` and ``G_2``.
"""
function SpmrScMatrix(A::T, G₁ᵀ::V, G₂::W) where {T<:InvLinearMap{Float64},
                                                  V<:RealOperator,
                                                  W<:RealOperator}
    if !checktranspose(A.map)
        error("Left-division by Aᵀ should be implemented")
    end

    Aᵀ = issymmetric(A.map) ? A : InvLinearMap(transpose(A.map))

    return SpmrScMatrix(A, Aᵀ, convert(FloatOperator, G₁ᵀ), convert(FloatOperator, G₂))
end

function SpmrScMatrix(A::T, G₁ᵀ::V, G₂::W) where {T<:AbstractMatrix{Float64},
                                                  V<:FloatOperator,
                                                  W<:FloatOperator}
    F = factorize(A)
    Fᵀ = issymmetric(A) ? F : factorize(copy(transpose(A)))

    return SpmrScMatrix(F, Fᵀ, G₁ᵀ, G₂)
end

"""
    SpmrScMatrix(A::AbstractMatrix{<:Real}, G₁ᵀ::RealOperator, G₂::RealOperator)

Construct an [`SpmrScMatrix`](@ref) from an `AbstractMatrix` representing ``A`` and
[`RealOperator`](@ref)s representing ``G_1^T`` and ``G_2``.
"""
function SpmrScMatrix(A::T, G₁ᵀ::V, G₂::W) where {T<:AbstractMatrix{<:Real},
                                                  V<:RealOperator,
                                                  W<:RealOperator}
    return SpmrScMatrix(convert(AbstractMatrix{Float64}, A),
                        convert(FloatOperator, G₁ᵀ),
                        convert(FloatOperator, G₂))
end

function SpmrScMatrix(K::AbstractMatrix{<:Real}, n::Integer)
    return SpmrScMatrix(K[1:n, 1:n], K[1:n, n+1:end], K[n+1:end, 1:n])
end

SpmrScMatrix(K::SpmrScMatrix) = K

function Base.Matrix(K::SpmrScMatrix)
    _, m = block_sizes(K)

    return [convert(Matrix, K.A) convert(Matrix, K.G₁ᵀ);
            convert(Matrix, K.G₂) zeros(m, m)]
end

function SparseArrays.sparse(K::SpmrScMatrix)
    _, m = block_sizes(K)

    try
        return [sparse(K.A) sparse(K.G₁ᵀ);
                sparse(K.G₂) spzeros(m, m)]
    catch
        return [sparse(convert(Matrix, K.A)) sparse(K.G₁ᵀ);
                sparse(K.G₂) spzeros(m, m)]
    end
end

Base.Array(K::SpmrScMatrix) = Matrix(K)
Base.convert(::Type{Matrix}, K::SpmrScMatrix) = Matrix(K)
Base.convert(::Type{Array}, K::SpmrScMatrix) = Array(K)
Base.convert(::Type{SparseMatrixCSC}, K::SpmrScMatrix) = sparse(K)

Base.size(K::SpmrScMatrix) = (size(K.A, 1) + size(K.G₂, 1),
                              size(K.A, 2) + size(K.G₁ᵀ, 2))

"""
    block_sizes(K::SpmrScMatrix)

Return ``(n, m)``.
"""
block_sizes(K::SpmrScMatrix) = (size(K.A, 1), size(K.G₂, 1))

# -NS family

"""
Saddle-point matrix type for SPMR-NS and SPQMR-NS representing a block matrix of the form
```math
\\begin{bmatrix}
A & G_1^T \\\\ G_2 & 0
\\end{bmatrix},
```
where ``A \\in \\mathbb{R}^{n \\times n}`` and ``G_1, G_2 \\in \\mathbb{R}^{m \\times n}``.
"""
struct SpmrNsMatrix{T<:FloatOperator,
                    U<:FloatOperator,
                    V<:FloatOperator} <: SpmrMatrix
    A::T
    H₁::U
    H₂::V

    m::Int

    function SpmrNsMatrix{T, U, V}(A, H₁, H₂, m) where {T<:FloatOperator,
                                                        U<:FloatOperator,
                                                        V<:FloatOperator}
        n = checksquare(A)

        if size(H₁, 1) ≠ n
            throw(DimensionMismatch("H₁ should have $n rows"))
        elseif size(H₂, 1) ≠ n
            throw(DimensionMismatch("H₂ should have $n rows"))
        end

        if size(H₁, 2) ≠ size(H₂, 2)
            throw(DimensionMismatch("H₁ and H₂ should have the same number of columns"))
        end

        if !checktranspose(A)
            error("Left-multiplication by Aᵀ should be implemented")
        elseif !checktranspose(H₁)
            error("Left-multiplication by H₁ᵀ should be implemented")
        elseif !checktranspose(H₂)
            error("Left-multiplication by H₂ᵀ should be implemented")
        end

        return new{T, U, V}(A, H₁, H₂, m)
    end
end

function SpmrNsMatrix(A::T, H₁::U, H₂::V, m::Int) where {T<:FloatOperator,
                                                         U<:FloatOperator,
                                                         V<:FloatOperator}
    return SpmrNsMatrix{T, U, V}(A, H₁, H₂, m)
end

"""
    SpmrNsMatrix(A::RealOperator, H₁::RealOperator, H₂::RealOperator, m::Integer)

Construct an [`SpmrNsMatrix`](@ref) from [`RealOperator`](@ref)s representing ``A``,
``H_1``, and ``H_2``, where ``H_1`` and ``H_2`` are nullspace bases for ``G_1`` and
``G_2``, respectively, and ``m`` is as above.
"""
function SpmrNsMatrix(A::T, H₁::U, H₂::V, m::Integer) where {T<:RealOperator,
                                                             U<:RealOperator,
                                                             V<:RealOperator}
    return SpmrNsMatrix(convert(FloatOperator, A),
                        convert(FloatOperator, H₁),
                        convert(FloatOperator, H₂),
                        convert(Int, m))
end

SpmrNsMatrix(K::SpmrNsMatrix) = K

"""
    block_sizes(K::SpmrNsMatrix)

Return ``(n, m)``.
"""
block_sizes(K::SpmrNsMatrix) = (size(K.A, 1), K.m)

"""
    nullsp_basis_size(K::SpmrNsMatrix)

Return the second dimension of ``H_1`` (and ``H_2``).

This is typically ``n-m`` or ``n``.
"""
nullsp_basis_size(K::SpmrNsMatrix) = size(K.H₁, 2)

# Helper functions

function checktranspose(A::RealOperator)
    return !(isa(A, FunctionMap) && !issymmetric(A) && A.fc == nothing)
end
