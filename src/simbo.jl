# SIMBO: Simultaneous bidiagonalization via bi-orthogonality

# SIMBO-SC

mutable struct SimboScIterator
    K::SpmrScMatrix
    SI::SpmrScIterate

    SI₀::SpmrScIterate  # Remember SI₀ so that we can reiterate over SSI
end

function simbo_sc(K::SpmrScMatrix, b::Vector{Float64}, c::Vector{Float64})
    n, m = block_sizes(K)

    χ = c ⋅ b

    β = sqrt(abs(χ))
    z = c / β
    δ = copysign(β, χ)
    v = b / δ

    # The rest is the same as in SIMBA-SC

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
    #println("v: $(v[1])")
    mul!(z, K.G₁ᵀ', SI_prev.w)
    BLAS.axpy!(-SI_prev.α, SI_prev.z, z)
    #println("z: $(z[1])")

    χ = z ⋅ v
    #println("χ: $χ")

    δ = sqrt(abs(χ))
    #println("δ: $δ")
    v ./= δ
    #println("v: $(v[1])")
    #BLAS.scal!(m, inv(δ), v, 1)
    #β = copysign(δ, χ)
    β = χ / δ
    #println("β: $β")
    z ./= β
    #println("z: $(z[1])")
    #BLAS.scal!(m, inv(β), z, 1)

    #println()
    # The rest is the same as in SIMBA-SC

    û = Vector{Float64}(undef, n)
    ŵ = Vector{Float64}(undef, n)

    mul!(û, K.G₁ᵀ, v)
    #println("û: $(û[1])")
    mul!(ŵ, K.G₂', z)
    #println("ŵ: $(ŵ[1])")

    u = copy(û)
    ldiv!(K.A, u)
    #println("$(SI_prev.ξ)")
    #println("$(u[1])")
    a = u - copysign(β, SI_prev.ξ) * SI_prev.u
    #println("u_prev: $(SI_prev.u[1])")
    #println("$(a[1])")
    BLAS.axpy!(-sign(SI_prev.ξ) * β, SI_prev.u, u)
    #println("u: $(u[1])")
    w = copy(ŵ)
    ldiv!(K.Aᵀ, w)
    BLAS.axpy!(-sign(SI_prev.ξ) * δ, SI_prev.w, w)
    #println("w: $(w[1])")

    ξ = û ⋅ w
    #println("ξ: $(ξ)")

    α = sqrt(abs(ξ))
    #println("α: $(α)")

    #BLAS.scal!(n, inv(α), u, 1)
    u ./= α
    #println("u: $(u[1])")
    #BLAS.scal!(n, inv(α), w, 1)
    w ./= α
    #println("w: $(w[1])")

    α = copysign(α, ξ)
    γ = α

    SSI.SI = SpmrScIterate(α, β, γ, δ, ξ, u, v, w, z)

    return (SSI.SI, k+1)
end

Base.eltype(::Type{SimboScIterator}) = SpmrScIterate

Base.length(SSI::SimboScIterator) = block_sizes(SSI.K)[2]
