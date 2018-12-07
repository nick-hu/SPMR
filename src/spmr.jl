# SPMR: Saddle-point minimum residual

export spmr_sc

function spmr_sc(K::SPMatrix, g::AbstractVector{<:Real};
                 tol::Float64=1e-6, maxit::Int=10)
    n, m = block_sizes(K)

    x, y = zeros(n), zeros(m)
    SSI, SI₀ = simba_sc(K, g, g)

    abs(SI₀.ξ) < eps() && return SPMRResult(x, y, OTHER, 0, Float64[])

    ρ̄, ϕ̄ = SI₀.γ, SI₀.δ
    d = SI₀.u

    T = zeros(m, 3)
    t, t_prev, t_prev2 = @view(T[:, 1]), @view(T[:, 2]), @view(T[:, 3])

    BLAS.blascopy!(m, SI₀.v, 1, t, 1)
    μ, ν = zero(Float64), zero(Float64)
    α_prev, ξ_prev, σ_prev = SI₀.α, SI₀.ξ, zero(Float64)

    resvec = Vector{Float64}(undef, min(m, maxit))

    for (k, SI) in enumerate(SSI)
        k > maxit && return SPMRResult(x, y, MAXIT_EXCEEDED, maxit, resvec[1:maxit])
        abs(SI.ξ) < eps() && return SPMRResult(x, y, OTHER, k-1, resvec[1:k-1])

        Ω, ρ = givens(ρ̄, SI.δ, 1, 2)    # ρ_k
        σ, ρ̄ = Ω.s * SI.γ, Ω.c * SI.γ   # σ_{k+1}, ρ̄_{k+1}
        ϕ, ϕ̄ = Ω.c * ϕ̄, -Ω.s * ϕ̄        # ϕ_k, ϕ̄_{k+1}

        BLAS.axpy!(ϕ/ρ, d, x)       # x_k
        BLAS.scal!(n, -σ/ρ, d, 1)
        d .+= SI.u                  # d_{k+1}

        λ = copysign(ρ * α_prev, ξ_prev)                # λ_k
        T[:, 1] .= (t - μ * t_prev - ν * t_prev2) ./ λ  # t_k
        BLAS.axpy!(-ϕ, t, y)                            # y_k

        T .= circshift(T, (0, 1))
        BLAS.blascopy!(m, SI.v, 1, t, 1)
        μ = copysign(ρ * SI.β, ξ_prev) + copysign(σ * SI.α, SI.ξ)   # μ_{k+1}
        ν = copysign(σ_prev * SI.β, ξ_prev)                         # ν_{k+1}

        α_prev, ξ_prev, σ_prev = SI.α, SI.ξ, σ

        resvec[k] = k == 1 ? Ω.s : resvec[k-1] * Ω.s     # Rel. res. norm
        resvec[k] < tol && return SPMRResult(x, y, CONVERGED, k, resvec[1:k])
    end

    return SPMRResult(x, y, MAXIT_EXCEEDED, m, resvec)
end
