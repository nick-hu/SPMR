# Macros for SIMBA/SIMBO iterations

macro normalize!(β, v, v̂, b, m)
    return quote
        $β = sqrt($v̂ ⋅ $b)
        $v = $b / $β
        BLAS.scal!($m, inv($β), $v̂, 1)
    end |> esc
end

macro normalize!(β, v, v̂, m)
    return quote
        $β = sqrt($v̂ ⋅ $v)
        BLAS.scal!($m, inv($β), $v, 1)
        BLAS.scal!($m, inv($β), $v̂, 1)
    end |> esc
end

macro mul_into!(v̂, M, v, n)
    return quote
        $v̂ = Vector{Float64}(undef, $n)
        mul!($v̂, $M, $v)
    end |> esc
end

macro ldiv_into!(v̂, M, v, n)
    return quote
        $v̂ = Vector{Float64}(undef, $n)
        ldiv!($v̂, $M, $v)
    end |> esc
end

macro scal_signinv!(v, α, ξ, n)
    return quote
        BLAS.scal!($n, flipsign(inv($α), $ξ), $v, 1)
    end |> esc
end

macro conjugate!(u, w, α, γ, ξ, û, n)
    return quote
        $ξ = $û ⋅ $w
        $α = $γ = sqrt(abs($ξ))

        @scal_signinv!($u, $α, $ξ, $n)
        @scal_signinv!($w, $γ, $ξ, $n)
    end |> esc
end

macro biorthogonalize!(z, ẑ, v, v̂, β, δ, c, b, n)
    return quote
        χ = $c ⋅ $v̂

        $δ = sqrt(abs(χ))
        $v = $b / $δ
        BLAS.scal!($n, inv($δ), $v̂, 1)

        $β = flipsign($δ, χ)
        $z = $c / $β
        BLAS.scal!($n, inv($β), $ẑ, 1)
    end |> esc
end

macro biorthogonalize!(z, ẑ, v, v̂, β, δ, n)
    return quote
        χ = $z ⋅ $v̂

        $δ = sqrt(abs(χ))
        BLAS.scal!($n, inv($δ), $v, 1)
        BLAS.scal!($n, inv($δ), $v̂, 1)

        $β = flipsign($δ, χ)
        BLAS.scal!($n, inv($β), $z, 1)
        BLAS.scal!($n, inv($β), $ẑ, 1)
    end |> esc
end
