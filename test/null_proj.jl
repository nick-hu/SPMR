# Orthogonal projection onto the nullspace of a matrix (used in -NS family methods)

function null_proj(G::AbstractMatrix{<:Real})
    m, n = size(G)

    M = [sparse(I, n, n) G'; G spzeros(m, m)]
    F = lu(M)

    function constraint_precond_soln(c::Vector{Float64})
        v = zeros(n+m)

        copyto!(v, c)
        ldiv!(F, v)

        return v[1:n]
    end

    return constraint_precond_soln
end
