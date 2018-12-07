# SIMBA: Simultaneous bidiagonalization via A-conjugacy

struct SimbaIterate
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

mutable struct SimbaScIterator
    K::SPMatrix
    SI::SimbaIterate

    SI₀::SimbaIterate  # Remember SI₀ so that we can reiterate over SSI
end

function simba_sc(K::SPMatrix, b::Vector{Float64}, c::Vector{Float64})
    n, m = block_sizes(K)

    β = BLAS.nrm2(m, b, 1)
    v = b / β
    δ = BLAS.nrm2(m, c, 1)
    z = c / δ

    û = Vector{Float64}(undef, n)
    ŵ = Vector{Float64}(undef, n)

    mul!(û, K.G₁ᵀ, v)
    mul!(ŵ, K.G₂', z)

    u = copy(û)
    ldiv!(K.A, u)
    w = copy(ŵ)
    ldiv!(K.Aᵀ, w)

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α

    BLAS.scal!(n, copysign(inv(α), ξ), u, 1)
    BLAS.scal!(n, copysign(inv(γ), ξ), w, 1)

    SI = SimbaIterate(α, β, γ, δ, ξ, u, v, w, z)

    return (SimbaScIterator(K, SI, SI), SI)
end

function simba_sc(K::SPMatrix, b::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    return simba_sc(K, convert(Vector{Float64}, b), convert(Vector{Float64}, c))
end

function Base.iterate(SSI::SimbaScIterator, k::Int=0)
    n, m = block_sizes(SSI.K)

    k == 0 && (SSI.SI = SSI.SI₀)
    k ≥ m && return nothing

    K, SI_prev = SSI.K, SSI.SI

    v = Vector{Float64}(undef, m)
    z = Vector{Float64}(undef, m)

    mul!(v, K.G₁ᵀ', SI_prev.w)
    BLAS.axpy!(-SI_prev.α, SI_prev.v, v)
    mul!(z, K.G₂, SI_prev.u)
    BLAS.axpy!(-SI_prev.γ, SI_prev.z, z)

    β = BLAS.nrm2(m, v, 1)
    normalize!(v)
    δ = BLAS.nrm2(m, z, 1)
    normalize!(z)

    û = Vector{Float64}(undef, n)
    ŵ = Vector{Float64}(undef, n)

    mul!(û, K.G₁ᵀ, v)
    mul!(ŵ, K.G₂', z)

    u = copy(û)
    ldiv!(K.A, u)
    BLAS.axpy!(-copysign(β, SI_prev.ξ), SI_prev.u, u)
    w = copy(ŵ)
    ldiv!(K.Aᵀ, w)
    BLAS.axpy!(-copysign(δ, SI_prev.ξ), SI_prev.w, w)

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α

    BLAS.scal!(n, copysign(inv(α), ξ), u, 1)
    BLAS.scal!(n, copysign(inv(γ), ξ), w, 1)

    SSI.SI = SimbaIterate(α, β, γ, δ, ξ, u, v, w, z)

    return (SSI.SI, k+1)
end

Base.eltype(::Type{SimbaScIterator}) = SimbaIterate

Base.length(SSI::SimbaScIterator) = block_sizes(SSI.K)[2]
