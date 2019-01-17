# SIMBO: Simultaneous bidiagonalization via biorthogonality

# SIMBO-SC

mutable struct SimboScIterator
    K::SpmrScMatrix
    SI::SpmrScIterate

    SI₀::SpmrScIterate  # Remember SI₀ so that we can reiterate over SSI
end

function simbo_sc(K::SpmrScMatrix, b::Vector{Float64}, c::Vector{Float64})
    n, m = block_sizes(K)

    χ = c ⋅ b

    δ = sqrt(abs(χ))
    v = b / δ
    β = flipsign(δ, χ)
    z = c / β

    û = Vector{Float64}(undef, n)
    ŵ = Vector{Float64}(undef, n)

    mul!(û, K.G₁ᵀ, v)
    mul!(ŵ, K.G₂', z)

    u = copy(û)
    ldiv!(K.A, u)
    w = copy(ŵ)
    ldiv!(K.Aᵀ, w)

    ξ = û ⋅ w

    α = flipsign(sqrt(abs(ξ)), ξ)
    γ = α

    BLAS.scal!(n, flipsign(inv(α), ξ), u, 1)
    BLAS.scal!(n, flipsign(inv(γ), ξ), w, 1)

    SI = SpmrScIterate(α, β, γ, δ, ξ, u, v, w, z)

    return (SimboScIterator(K, SI, SI), SI)
end

function simbo_sc(K::SpmrScMatrix, b::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    return simbo_sc(K, convert(Vector{Float64}, b), convert(Vector{Float64}, c))
end

function Base.iterate(SSI::SimboScIterator, k::Int=0)
    n, m = block_sizes(SSI.K)

    k == 0 && (SSI.SI = SSI.SI₀)
    k ≥ m && return nothing

    K, SI_prev = SSI.K, SSI.SI

    v = Vector{Float64}(undef, m)
    z = Vector{Float64}(undef, m)

    mul!(v, K.G₂, SI_prev.u)
    BLAS.axpy!(-SI_prev.γ, SI_prev.v, v)
    mul!(z, K.G₁ᵀ', SI_prev.w)
    BLAS.axpy!(-SI_prev.α, SI_prev.z, z)

    χ = z ⋅ v

    δ = sqrt(abs(χ))
    BLAS.scal!(m, inv(δ), v, 1)
    β = flipsign(δ, χ)
    BLAS.scal!(m, inv(β), z, 1)

    û = Vector{Float64}(undef, n)
    ŵ = Vector{Float64}(undef, n)

    mul!(û, K.G₁ᵀ, v)
    mul!(ŵ, K.G₂', z)

    u = copy(û)
    ldiv!(K.A, u)
    BLAS.axpy!(-flipsign(β, SI_prev.ξ), SI_prev.u, u)
    w = copy(ŵ)
    ldiv!(K.Aᵀ, w)
    BLAS.axpy!(-flipsign(δ, SI_prev.ξ), SI_prev.w, w)

    ξ = û ⋅ w

    α = flipsign(sqrt(abs(ξ)), ξ)
    γ = α

    BLAS.scal!(n, flipsign(inv(α), ξ), u, 1)
    BLAS.scal!(n, flipsign(inv(γ), ξ), w, 1)

    SSI.SI = SpmrScIterate(α, β, γ, δ, ξ, u, v, w, z)

    return (SSI.SI, k+1)
end

Base.eltype(::Type{SimboScIterator}) = SpmrScIterate

Base.length(SSI::SimboScIterator) = block_sizes(SSI.K)[2]
