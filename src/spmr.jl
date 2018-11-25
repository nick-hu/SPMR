# SPMR: Saddle-point minimum residual

export spmr_sc

function spmr_sc(K::SPMatrix, g::AbstractVector{<:Real};
                 tol::Float64=1e-6, maxit::Int=10)
    n, m = block_sizes(K)

    x, y = zeros(n), zeros(m)
    SSI = simba_sc(K, g, g)

    abs(SSI.SI₀.ξ) < eps() && return SPMRResult(x, y, OTHER, SSI.SI₀.k, Float64[])

    ρ̄, ϕ̄ = SSI.SI₀.γ, SSI.SI₀.δ
    d = SSI.SI₀.u

    T = Matrix{Float64}(undef, m, 3)
    BLAS.blascopy!(m, SSI.SI₀.v, 1, T, 1)
    μ, ν = zero(Float64), zero(Float64)
    α_prev, ξ_prev, σ_prev = SSI.SI₀.α, SSI.SI₀.ξ, zero(Float64)

    resvec = zeros(min(m, maxit))

    for SI in SSI
        SI.k > maxit && return SPMRResult(x, y, MAXIT_EXCEEDED, maxit, resvec[1:maxit])
        abs(SI.ξ) < eps() && return SPMRResult(x, y, OTHER, SI.k - 1, resvec[1:SI.k - 1])

        Ω, ρ = givens(ρ̄, SI.δ, 1, 2)    # ρ_k
        σ, ρ̄ = Ω * [zero(SI.γ), SI.γ]   # σ_{k+1}, ρ̄_{k+1}
        ϕ, ϕ̄ = Ω * [ϕ̄, zero(ϕ̄)]         # ϕ_k, ϕ̄_{k+1}

        BLAS.axpy!(ϕ/ρ, d, x)   # x_k
        BLAS.scal!(n, -σ/ρ, d, 1)
        BLAS.axpy!(1, SI.u, d)  # d_{k+1}

        λ = copysign(ρ * α_prev, ξ_prev)                        # λ_k
        T[:, 1] .= (T[:, 1] - μ * T[:, 2] - ν * T[:, 3]) / λ    # t_k
        BLAS.axpy!(-ϕ, T[:, 1], y)                              # y_k

        T .= circshift(T, (0, 1))
        BLAS.blascopy!(m, SI.v, 1, T, 1)
        μ = copysign(ρ * SI.β, ξ_prev) + copysign(σ * SI.α, SI.ξ)   # μ_{k+1}
        ν = copysign(σ_prev * SI.β, ξ_prev)                         # ν_{k+1}

        α_prev, ξ_prev, σ_prev = SI.α, SI.ξ, σ

        resvec[SI.k] = SI.k == 1 ? Ω.s : resvec[SI.k - 1] * Ω.s     # Rel. res. norm
        resvec[SI.k] < tol && return SPMRResult(x, y, CONVERGED, SI.k, resvec[1:SI.k])
    end

    return SPMRResult(x, y, CONVERGED, m, resvec)
end
