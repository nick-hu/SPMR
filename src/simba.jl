# SIMBA: Simultaneous bidiagonalization via A-conjugacy

function simba_sc(A::Matrix{Float64}, G₁::Matrix{Float64}, G₂::Matrix{Float64},
                  b::Vector{Float64}, c::Vector{Float64})
    return 1
end

function simba_sc(A::Matrix{<:Real}, G₁::Matrix{<:Real}, G₂::Matrix{<:Real},
                  b::Vector{<:Real}, c::Vector{<:Real})

    return simba_sc(convert(Matrix{Float64}, A),
                    convert(Matrix{Float64}, G₁),
                    convert(Matrix{Float64}, G₂),
                    convert(Vector{Float64}, b),
                    convert(Vector{Float64}, c))
end
