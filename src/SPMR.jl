module SPMR

using LinearAlgebra: checksquare

export
    SPMatrix,

    simba_sc

include("saddlepoint.jl")

include("simba.jl")

end # module
