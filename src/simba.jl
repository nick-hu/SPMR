# SIMBA: Simultaneous bidiagonalization via A-conjugacy

struct SimbaScIterator{U<:FloatMatrix, V<:FloatMatrix}
    A::Factorization{Float64}
    G₁ᵀ::U
    G₂::V

    b::Vector{Float64}
    c::Vector{Float64}
end

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

SimbaIterate() = SimbaIterate(0, Vector{Float64}(undef, 5)..., map(_ -> Float64[], 1:4)...)

function simba_sc(K::SPMatrix, b::Vector{Float64}, c::Vector{Float64})
    return SimbaScIterator(factorize(K.A), K.G₁ᵀ, K.G₂, b, c)
end

function simba_sc(K::SPMatrix, b::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    return simba_sc(K, convert(Vector{Float64}, b), convert(Vector{Float64}, c))
end

function Base.iterate(SSI::SimbaScIterator, SI_prev::SimbaIterate=SimbaIterate())
    n, m = size(SSI.A, 1), size(SSI.G₂, 1)
    k = SI_prev.k

    if k ≥ m
        return nothing
    end

    if k == 0
        β = norm(SSI.b)
        v = SSI.b / β
        δ = norm(SSI.c)
        z = SSI.c / δ

        û = SSI.G₁ᵀ * v
        ŵ = SSI.G₂' * z
        u = SSI.A \ û
        w = SSI.A' \ ŵ
    else
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
    end

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α
    u *= copysign(inv(α), ξ)
    w *= copysign(inv(γ), ξ)

    SI = SimbaIterate(k+1, α, β, γ, δ, ξ, u, v, w, z)

    return (SI, SI)
end

Base.eltype(::Type{SimbaScIterator{U, V}}) where {U, V} = SimbaIterate

Base.length(SSI::SimbaScIterator) = size(SSI.G₂, 1)
