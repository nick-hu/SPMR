# SIMBA: Simultaneous bidiagonalization via A-conjugacy

struct SimbaIterate
    k::Int

    α::Float64
    β::Float64
    γ::Float64
    δ::Float64

    ξ::Float64

    u::Vector{Float64}
    v::Vector{Float64}
    w::Vector{Float64}
    z::Vector{Float64}
end

struct SimbaScIterator{T<:Factorization{Float64}, U<:FloatMatrix, V<:FloatMatrix}
    A::T
    Aᵀ::T
    G₁ᵀ::U
    G₂::V

    SI₀::SimbaIterate
end

function simba_sc(K::SPMatrix, b::Vector{Float64}, c::Vector{Float64})
    n, m = block_sizes(K)

    A = factorize(K.A)
    Aᵀ = issymmetric(K.A) ? A : factorize(copy(K.A'))

    β = BLAS.nrm2(m, b, 1)
    v = b / β
    δ = BLAS.nrm2(m, c, 1)
    z = c / δ

    û = Vector{Float64}(undef, n)
    ŵ = Vector{Float64}(undef, n)

    mul!(û, K.G₁ᵀ, v)
    mul!(ŵ, K.G₂', z)

    u = copy(û)
    ldiv!(A, u)
    w = copy(ŵ)
    ldiv!(Aᵀ, w)

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α

    BLAS.scal!(n, copysign(inv(α), ξ), u, 1)
    BLAS.scal!(n, copysign(inv(γ), ξ), w, 1)

    SI₀ = SimbaIterate(0, α, β, γ, δ, ξ, u, v, w, z)

    return SimbaScIterator(A, Aᵀ, K.G₁ᵀ, K.G₂, SI₀)
end

function simba_sc(K::SPMatrix, b::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    return simba_sc(K, convert(Vector{Float64}, b), convert(Vector{Float64}, c))
end

function Base.iterate(SSI::SimbaScIterator, SI_prev::SimbaIterate=SSI.SI₀)
    n, m = block_sizes(SSI)

    if SI_prev.k ≥ m
        return nothing
    end

    v = Vector{Float64}(undef, m)
    z = Vector{Float64}(undef, m)

    mul!(v, SSI.G₁ᵀ', SI_prev.w)
    BLAS.axpy!(-SI_prev.α, SI_prev.v, v)
    mul!(z, SSI.G₂, SI_prev.u)
    BLAS.axpy!(-SI_prev.γ, SI_prev.z, z)

    β = BLAS.nrm2(m, v, 1)
    BLAS.scal!(m, inv(β), v, 1)
    δ = BLAS.nrm2(m, z, 1)
    BLAS.scal!(m, inv(δ), z, 1)

    û = Vector{Float64}(undef, n)
    ŵ = Vector{Float64}(undef, n)

    mul!(û, SSI.G₁ᵀ, v)
    mul!(ŵ, SSI.G₂', z)

    u = copy(û)
    ldiv!(SSI.A, u)
    BLAS.axpy!(-copysign(β, SI_prev.ξ), SI_prev.u, u)
    w = copy(ŵ)
    ldiv!(SSI.Aᵀ, w)
    BLAS.axpy!(-copysign(δ, SI_prev.ξ), SI_prev.w, w)

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α

    BLAS.scal!(n, copysign(inv(α), ξ), u, 1)
    BLAS.scal!(n, copysign(inv(γ), ξ), w, 1)

    SI = SimbaIterate(SI_prev.k + 1, α, β, γ, δ, ξ, u, v, w, z)

    return (SI, SI)
end

Base.eltype(::Type{SimbaScIterator{T, U, V}}) where {T, U, V} = SimbaIterate

Base.length(SSI::SimbaScIterator) = block_sizes(SSI)[2]

block_sizes(SSI::SimbaScIterator) = (size(SSI.A, 1), size(SSI.G₂, 1))
