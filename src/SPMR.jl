module SPMR

using LinearAlgebra
using LinearAlgebra: checksquare

export
    SPMatrix,

    simba_sc,

    spmr_sc

const FloatMatrix = AbstractMatrix{Float64}
const RealMatrix = AbstractMatrix{<:Real}
const FloatFact = Factorization{Float64}
const RealFact = Factorization{<:Real}

undef_vec(n::Int) = Vector{Float64}(undef, n)

include("saddlepoint.jl")

include("simba.jl")

include("spmr.jl")

end # module
