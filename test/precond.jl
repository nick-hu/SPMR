using SPMR: InvLinearMap

function constraint_precond(Ã::AbstractMatrix{<:Real}, G::AbstractMatrix{<:Real})
    n = size(A, 1)
    m = size(G, 1)

    M = [Ã G'; G spzeros(m, m)]
    F = lu(M)

    function constraint_precond_soln(c::Vector{Float64})
        v = zeros(n+m)

        copyto!(v, c)
        ldiv!(F, v)

        return v[1:n]
    end

    return InvLinearMap(constraint_precond_soln, n)
end
