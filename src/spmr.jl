# SPMR: Saddle-point minimal residual

export
    spmr_sc, spmr_ns,
    spqmr_sc, spqmr_ns,
    recover_y

include("spmr_macros.jl")

@enum SpmrFlag begin
    CONVERGED
    MAXIT_EXCEEDED
    OTHER
end

struct SpmrScResult
    x::Vector{Float64}
    y::Vector{Float64}
    flag::SpmrFlag
    iter::Int
    resvec::Vector{Float64}
end

struct SpmrNsResult
    x::Vector{Float64}
    flag::SpmrFlag
    iter::Int
    resvec::Vector{Float64}
end

# SPMR-SC

function spmr_sc(K::SpmrScMatrix, g::AbstractVector{<:Real};
                 tol::Float64=1e-6, maxit::Int=10)
    n, m = block_sizes(K)
    SSI, SI₀ = simba_sc(K, g, g)

    @init_qr(SI₀)
    @init_x!(x, SI₀, n)
    @init_y!(y, SI₀, m)

    abs(SI₀.ξ) < eps() && return SpmrScResult(x, y, OTHER, 0, Float64[])

    resvec = Vector{Float64}(undef, min(m, maxit))

    for (k, SI) in enumerate(SSI)
        k > maxit && return SpmrScResult(x, y, MAXIT_EXCEEDED, maxit, resvec[1:maxit])
        abs(SI.ξ) < eps() && return SpmrScResult(x, y, OTHER, k-1, resvec[1:k-1])

        @update_qr!(Ω, SI)
        @update_x!(x, SI, n)
        @update_y!(y, SI, m)

        resvec[k] = k == 1 ? Ω.s : resvec[k-1] * Ω.s    # Rel. res. norm
        resvec[k] < tol && return SpmrScResult(x, y, CONVERGED, k, resvec[1:k])
    end

    return SpmrScResult(x, y, MAXIT_EXCEEDED, m, resvec)
end

# SPMR-NS

function spmr_ns(K::SpmrNsMatrix, f::AbstractVector{<:Real};
                 tol::Float64=1e-6, maxit::Int=10)
    n, m = block_sizes(K)

    fₕ = K.H₁' * -f
    SNI, SI₀ = simbo_ns(K, fₕ, fₕ)

    @init_qr(SI₀)
    @init_x!(p, SI₀, n)

    abs(SI₀.ξ) < eps() && return SpmrNsResult(-p, OTHER, 0, Float64[])

    resvec = Vector{Float64}(undef, min(n-m, maxit))

    for (k, SI) in enumerate(SNI)
        k > maxit && return SpmrNsResult(-p, MAXIT_EXCEEDED, maxit, resvec[1:maxit])
        abs(SI.ξ) < eps() && return SpmrNsResult(-p, OTHER, k-1, resvec[1:k-1])

        @update_qr!(Ω, SI)
        @update_x!(p, SI, n)

        resvec[k] = k == 1 ? Ω.s : resvec[k-1] * Ω.s     # Rel. res. norm
        resvec[k] < tol && return SpmrNsResult(-p, CONVERGED, k, resvec[1:k])
    end

    return SpmrNsResult(-p, MAXIT_EXCEEDED, m, resvec)
end

# SPQMR-SC

function spqmr_sc(K::SpmrScMatrix, g::AbstractVector{<:Real};
                  tol::Float64=1e-6, maxit::Int=10)
    n, m = block_sizes(K)
    SSI, SI₀ = simbo_sc(K, g, g)

    @init_qr(SI₀)
    @init_x!(x, SI₀, n)
    @init_y!(y, SI₀, m)

    abs(SI₀.ξ) < eps() && return SpmrScResult(x, y, OTHER, 0, Float64[])

    resvec = Vector{Float64}(undef, min(m, maxit))
    relres = one(Float64)
    norm_g = BLAS.nrm2(m, g, 1)

    for (k, SI) in enumerate(SSI)
        k > maxit && return SpmrScResult(x, y, MAXIT_EXCEEDED, maxit, resvec[1:maxit])
        abs(SI.ξ) < eps() && return SpmrScResult(x, y, OTHER, k-1, resvec[1:k-1])

        @update_qr!(Ω, SI)
        @update_x!(x, SI, n)
        @update_y!(y, SI, m)

        relres *= Ω.s
        resvec[k] = sqrt(k) * relres    # Bound on rel. res. norm

        if resvec[k] < tol
            r = Vector{Float64}(undef, m)
            mul!(r, K.G₂, x)
            r .-= g

            norm_r = BLAS.nrm2(m, r, 1)     # Actual residual norm
            norm_r < tol * norm_g && return SpmrScResult(x, y, CONVERGED, k, resvec[1:k])
        end
    end

    return SpmrScResult(x, y, MAXIT_EXCEEDED, m, resvec)
end

# SPQMR-NS

function spqmr_ns(K::SpmrNsMatrix, f::AbstractVector{<:Real};
                  tol::Float64=1e-6, maxit::Int=10)
    n, m = block_sizes(K)
    ℓ = nullsp_basis_size(K)

    fₕ = K.H₁' * -f
    SNI, SI₀ = simbo_ns(K, fₕ, fₕ)

    @init_qr(SI₀)
    @init_x!(p, SI₀, n)

    abs(SI₀.ξ) < eps() && return SpmrNsResult(-p, OTHER, 0, Float64[])

    resvec = Vector{Float64}(undef, min(n-m, maxit))
    relres = one(Float64)
    norm_f = BLAS.nrm2(n, f, 1)

    for (k, SI) in enumerate(SNI)
        k > maxit && return SpmrNsResult(-p, MAXIT_EXCEEDED, maxit, resvec[1:maxit])
        abs(SI.ξ) < eps() && return SpmrNsResult(-p, OTHER, k-1, resvec[1:k-1])

        @update_qr!(Ω, SI)
        @update_x!(p, SI, n)

        relres *= Ω.s
        resvec[k] = sqrt(k) * relres    # Bound on rel. res. norm

        if resvec[k] < tol
            pₐ = Vector{Float64}(undef, n)
            r = Vector{Float64}(undef, ℓ₁)
            mul!(pₐ, K.A, p)
            mul!(r, K.H₁', pₐ)
            r .-= fₕ

            norm_r = BLAS.nrm2(ℓ₁, r, 1)     # Actual residual norm
            norm_r < tol * norm_f && return SpmrNsResult(-p, CONVERGED, k, resvec[1:k])
        end
    end

    return SpmrNsResult(-p, MAXIT_EXCEEDED, m, resvec)
end

function recover_y(result::SpmrNsResult, K::SpmrNsMatrix, G₁ᵀ::AbstractMatrix{<:Real},
                   f::Vector{<:Real})
    Ax = Vector{Float64}(undef, block_sizes(K)[1])
    mul!(Ax, K.A, result.x)

    return G₁ᵀ \ (f - Ax)
end
