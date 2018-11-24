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
    G₁ᵀ::U
    G₂::V

    SI₀::SimbaIterate
end

function simba_sc(K::SPMatrix, b::Vector{Float64}, c::Vector{Float64})
    A = factorize(K.A)

    β = norm(b)
    v = b / β
    δ = norm(c)
    z = c / δ

    û = K.G₁ᵀ * v
    ŵ = K.G₂' * z
    u = (A \ û)::Vector{Float64}  # Annotate since factorization type is unknown
    w = (A' \ ŵ)::Vector{Float64}  # Annotate since factorization type is unknown

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α
    u *= copysign(inv(α), ξ)
    w *= copysign(inv(γ), ξ)

    SI₀ = SimbaIterate(0, α, β, γ, δ, ξ, u, v, w, z)

    return SimbaScIterator(A, K.G₁ᵀ, K.G₂, SI₀)
end

function simba_sc(K::SPMatrix, b::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    return simba_sc(K, convert(Vector{Float64}, b), convert(Vector{Float64}, c))
end

function Base.iterate(SSI::SimbaScIterator, SI_prev::SimbaIterate=SSI.SI₀)
    if SI_prev.k ≥ size(SSI.G₂, 1) - 1
        return nothing
    end

    v = SSI.G₁ᵀ' * SI_prev.w - SI_prev.α * SI_prev.v
    β = norm(v)
    v /= β

    z = SSI.G₂ * SI_prev.u - SI_prev.γ * SI_prev.z
    δ = norm(z)
    z /= δ

    û = SSI.G₁ᵀ * v
    ŵ = SSI.G₂' * z
    u = SSI.A \ û - copysign(β, SI_prev.ξ) * SI_prev.u
    w = SSI.A' \ ŵ - copysign(δ, SI_prev.ξ) * SI_prev.w

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α
    u *= copysign(inv(α), ξ)
    w *= copysign(inv(γ), ξ)

    SI = SimbaIterate(SI_prev.k + 1, α, β, γ, δ, ξ, u, v, w, z)

    return (SI, SI)
end

Base.eltype(::Type{SimbaScIterator{T, U, V}}) where {T, U, V} = SimbaIterate

Base.length(SSI::SimbaScIterator) = size(SSI.G₂, 1) - 1
