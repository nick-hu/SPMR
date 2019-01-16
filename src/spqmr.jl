# SPQMR: Saddle-point quasi-minimal residual

export
    spqmr_sc, spqmr_ns

# SPQMR-SC

function spqmr_sc(K::SpmrScMatrix, g::AbstractVector{<:Real};
                  tol::Float64=1e-6, maxit::Int=10)
    n, m = block_sizes(K)

    x, y = zeros(n), zeros(m)
    SSI, SI₀ = simbo_sc(K, g, g)

    abs(SI₀.ξ) < eps() && return SpmrScResult(x, y, OTHER, 0, Float64[])

    ρ̄, ϕ̄ = SI₀.γ, SI₀.δ
    d = SI₀.u

    T = zeros(m, 3)
    t, t_prev, t_prev2 = @view(T[:, 1]), @view(T[:, 2]), @view(T[:, 3])

    BLAS.blascopy!(m, SI₀.v, 1, t, 1)
    μ, ν = zero(Float64), zero(Float64)
    α_prev, ξ_prev, σ_prev = SI₀.α, SI₀.ξ, zero(Float64)

    resvec = Vector{Float64}(undef, min(m, maxit))
    norm_g = BLAS.nrm2(m, g, 1)

    for (k, SI) in enumerate(SSI)
    #println()

        k > maxit && return SpmrScResult(x, y, MAXIT_EXCEEDED, maxit, resvec[1:maxit])
        abs(SI.ξ) < eps() && return SpmrScResult(x, y, OTHER, k-1, resvec[1:k-1])

        Ω, ρ = givens(ρ̄, SI.δ, 1, 2)    # ρ_k
        σ, ρ̄ = Ω.s * SI.γ, Ω.c * SI.γ   # σ_{k+1}, ρ̄_{k+1}
        ϕ, ϕ̄ = Ω.c * ϕ̄, -Ω.s * ϕ̄        # ϕ_k, ϕ̄_{k+1}

        BLAS.axpy!(ϕ/ρ, d, x)       # x_k
        BLAS.scal!(n, -σ/ρ, d, 1)
        d .+= SI.u                  # d_{k+1}

        λ = ρ * α_prev * sign(ξ_prev)                # λ_k
        T[:, 1] .= (t - μ * t_prev - ν * t_prev2) ./ λ  # t_k
        BLAS.axpy!(-ϕ, t, y)                            # y_k

        T .= circshift(T, (0, 1))
        BLAS.blascopy!(m, SI.v, 1, t, 1)
        μ = (ρ * SI.β) * sign(ξ_prev) + (σ * SI.α) * sign(SI.ξ)   # μ_{k+1}
        ν = (σ_prev * SI.β) * sign(ξ_prev)                         # ν_{k+1}

        α_prev, ξ_prev, σ_prev = SI.α, SI.ξ, σ

        resvec[k] = k == 1 ? Ω.s : sqrt(k) * resvec[k-1] * Ω.s      # Bound on rel. res. norm
        if resvec[k] < tol
            r = Vector{Float64}(undef, m)
            mul!(r, K.G₂, x)
            r .-= g     # Actual residual

            norm_r = BLAS.nrm2(m, r, 1)
            norm_r < tol * norm_g && return SpmrScResult(x, y, CONVERGED, k, resvec[1:k])
        end
    end

    return SpmrScResult(x, y, MAXIT_EXCEEDED, m, resvec)
end

function spqmr_ns
end
