# SIMBA: Simultaneous bidiagonalization via A-conjugacy

# SIMBA-SC

mutable struct SimbaScIterator
    K::SpmrScMatrix
    SI::SpmrScIterate

    SI₀::SpmrScIterate  # Remember SI₀ so that we can reiterate over SSI
end

function simba_sc(K::SpmrScMatrix, b::Vector{Float64}, c::Vector{Float64})
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

    SI = SpmrScIterate(α, β, γ, δ, ξ, u, v, w, z)

    return (SimbaScIterator(K, SI, SI), SI)
end

function simba_sc(K::SpmrScMatrix, b::AbstractVector{<:Real}, c::AbstractVector{<:Real})
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

    SSI.SI = SpmrScIterate(α, β, γ, δ, ξ, u, v, w, z)

    return (SSI.SI, k+1)
end

Base.eltype(::Type{SimbaScIterator}) = SpmrScIterate

Base.length(SSI::SimbaScIterator) = block_sizes(SSI.K)[2]

# SIMBA-NS

mutable struct SimbaNsIterator
    K::SpmrNsMatrix
    SI::SpmrNsIterate

    SI₀::SpmrNsIterate  # Remember SI₀ so that we can reiterate over SNI
end

function simba_ns(K::SpmrNsMatrix, b::Vector{Float64}, c::Vector{Float64})
    n, _ = block_sizes(K)
    ℓ₁, ℓ₂ = nullities(K)

    β = BLAS.nrm2(ℓ₂, b, 1)
    v = b / β
    δ = BLAS.nrm2(ℓ₁, c, 1)
    z = c / δ

    u = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    mul!(u, K.H₂, v)
    mul!(w, K.H₁, z)

    û = copy(u)
    û .= K.A * û
    #lmul!(K.A, u)
    ŵ = copy(w)
    #lmul!(K.A', w)
    ŵ .= K.A' * ŵ

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α

    BLAS.scal!(n, copysign(inv(α), ξ), u, 1)
    BLAS.scal!(n, copysign(inv(γ), ξ), w, 1)

    SI = SpmrNsIterate(α, β, γ, δ, ξ, u, v, w, z, û, ŵ)

    return (SimbaNsIterator(K, SI, SI), SI)
end

function simba_ns(K::SpmrNsMatrix, b::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    return simba_ns(K, convert(Vector{Float64}, b), convert(Vector{Float64}, c))
end

function Base.iterate(SNI::SimbaNsIterator, k::Int=0)
    n, m = block_sizes(SNI.K)
    ℓ₁, ℓ₂ = nullities(SNI.K)

    k == 0 && (SNI.SI = SNI.SI₀)
    k ≥ n-m && return nothing

    K, SI_prev = SNI.K, SNI.SI

    v = Vector{Float64}(undef, ℓ₂)
    z = Vector{Float64}(undef, ℓ₁)

    BLAS.scal!(n, inv(SI_prev.γ), SI_prev.ŵ, 1)
    BLAS.scal!(n, inv(SI_prev.α), SI_prev.û, 1)

    mul!(v, K.H₂', SI_prev.ŵ)
    BLAS.axpy!(-SI_prev.α, SI_prev.v, v)
    mul!(z, K.H₁', SI_prev.û)
    BLAS.axpy!(-SI_prev.γ, SI_prev.z, z)

    β = BLAS.nrm2(ℓ₂, v, 1)
    normalize!(v)
    δ = BLAS.nrm2(ℓ₁, z, 1)
    normalize!(z)

    u = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    mul!(u, K.H₂, v)
    BLAS.axpy!(-copysign(β, SI_prev.ξ), SI_prev.u, u)
    mul!(w, K.H₁, z)
    BLAS.axpy!(-copysign(δ, SI_prev.ξ), SI_prev.w, w)

    û = copy(u)
    #lmul!(K.A, u)
    û .= K.A * û
    ŵ = copy(w)
    #lmul!(K.A', w)
    ŵ .= K.A' * ŵ

    ξ = û ⋅ w

    α = sqrt(abs(ξ))
    γ = α

    BLAS.scal!(n, copysign(inv(α), ξ), u, 1)
    BLAS.scal!(n, copysign(inv(γ), ξ), w, 1)

    SNI.SI = SpmrNsIterate(α, β, γ, δ, ξ, u, v, w, z, û, ŵ)

    return (SNI.SI, k+1)
end

Base.eltype(::Type{SimbaNsIterator}) = SpmrNsIterate

Base.length(SNI::SimbaNsIterator) = block_sizes(SNI.K)[2]
