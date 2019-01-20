# Macros for SIMBA/SIMBO iterations

macro normalize!(β, v, b, m)
    return quote
        $β = BLAS.nrm2($m, $b, 1)
        $v = $b / $β
    end |> esc
end

macro normalize!(β, v, m)
    return quote
        $β = BLAS.nrm2($m, $v, 1)
        normalize!($v)
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
