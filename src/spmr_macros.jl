# Macros for SPMR/SPQMR iterations

macro init_qr(SI₀)
    return quote
        ρ̄, ϕ̄ = $(SI₀).γ, $(SI₀).δ
    end |> esc
end

macro update_qr!(Ω, SI)
    return quote
        $(Ω), ρ = givens(ρ̄, $(SI).δ, 1, 2)          # ρ_k
        σ, ρ̄ = $(Ω).s * $(SI).γ, $(Ω).c * $(SI).γ   # σ_{k+1}, ρ̄_{k+1}
        ϕ, ϕ̄ = $(Ω).c * ϕ̄, -$(Ω).s * ϕ̄              # ϕ_k, ϕ̄_{k+1}
    end |> esc
end

macro init_x!(x, SI₀, n)
    return quote
        $x = zeros($n)
        d = $(SI₀).u
    end |> esc
end

macro update_x!(x, SI, n)
    return quote
        BLAS.axpy!(ϕ/ρ, d, $x)      # x_k
        BLAS.scal!($n, -σ/ρ, d, 1)
        d .+= $(SI).u               # d_{k+1}
    end |> esc
end

macro init_y!(y, SI₀, m)
    return quote
        $y = zeros($m)

        T = zeros($m, 3)
        t, t_prev, t_prev2 = @view(T[:, 1]), @view(T[:, 2]), @view(T[:, 3])

        BLAS.blascopy!($m, $(SI₀).v, 1, t, 1)
        μ, ν = zero(Float64), zero(Float64)
        α_prev, ξ_prev, σ_prev = $(SI₀).α, $(SI₀).ξ, zero(Float64)
    end |> esc
end

macro update_y!(y, SI, m)
    return quote
        λ = flipsign(ρ * α_prev, ξ_prev)                # λ_k
        T[:, 1] .= (t - μ * t_prev - ν * t_prev2) ./ λ  # t_k
        BLAS.axpy!(-ϕ, t, $y)                           # y_k

        T .= circshift(T, (0, 1))
        BLAS.blascopy!($m, $(SI).v, 1, t, 1)
        μ = flipsign(ρ * $(SI).β, ξ_prev) + flipsign(σ * $(SI).α, $(SI).ξ)  # μ_{k+1}
        ν = flipsign(σ_prev * $(SI).β, ξ_prev)                              # ν_{k+1}

        α_prev, ξ_prev, σ_prev = SI.α, SI.ξ, σ
    end |> esc
end
