# Grcar matrix

function grcar(n::Int, k::Int=3)
    A = spzeros(Int, n, n)

    @inbounds A[diagind(A, -1)] .= -1

    @inbounds for i in 0:k
        A[diagind(A, i)] .= 1
    end

    return A
end
