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
    Aᵀ = factorize(copy(K.A'))

    β = BLAS.nrm2(m, b, 1)
    v = b / β
    δ = BLAS.nrm2(m, c, 1)
    z = c / δ

    û = BLAS.gemv('N', K.G₁ᵀ, v)
    ŵ = BLAS.gemv('T', K.G₂, z)
    u = (A \ û)::Vector{Float64}  # Annotate since factorization type is unknown
    w = (Aᵀ \ ŵ)::Vector{Float64}  # Annotate since factorization type is unknown

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

    v = copy(SI_prev.v)
    BLAS.gemv!('T', one(Float64), SSI.G₁ᵀ, SI_prev.w, -SI_prev.α, v)
    β = BLAS.nrm2(m, v, 1)
    BLAS.scal!(m, inv(β), v, 1)

    z = copy(SI_prev.z)
    BLAS.gemv!('N', one(Float64), SSI.G₂, SI_prev.u, -SI_prev.γ, z)
    δ = BLAS.nrm2(m, z, 1)
    BLAS.scal!(m, inv(δ), z, 1)

    û = BLAS.gemv('N', SSI.G₁ᵀ, v)
    ŵ = BLAS.gemv('T', SSI.G₂, z)
    u = SSI.A \ û - copysign(β, SI_prev.ξ) * SI_prev.u
    w = SSI.Aᵀ \ ŵ - copysign(δ, SI_prev.ξ) * SI_prev.w

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α
    BLAS.scal!(n, copysign(inv(α), ξ), u, 1)
    BLAS.scal!(n, copysign(inv(γ), ξ), w, 1)

    SI = SimbaIterate(SI_prev.k + 1, α, β, γ, δ, ξ, u, v, w, z)

    return (SI, SI)
end

Base.eltype(::Type{SimbaScIterator{T, U, V}}) where {T, U, V} = SimbaIterate

Base.length(SSI::SimbaScIterator) = size(SSI.G₂, 1)

block_sizes(SSI::SimbaScIterator) = (size(SSI.A, 1), size(SSI.G₂, 1))
