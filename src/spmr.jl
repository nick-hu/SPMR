# SPMR: Saddle-point minimum residual

function spmr_sc(K::SPMatrix, g::AbstractVector{<:Real};
                 tol::Float64=1e-6, maxit::Int=10)
    SSI = simba_sc(K, g, g)

    ρ̄ = SSI.SI₀.γ
    ϕ̄ = SSI.SI₀.δ

    for SI in SSI
        if SI.k > maxit
            break
        end

        Ω, ρ = givens(ρ̄, SI.δ, 1, 2)
        σ, ρ̄ = Ω * [0, SI.γ]
        ϕ, ϕ̄ = Ω * [ϕ̄, 0]
    end
end
