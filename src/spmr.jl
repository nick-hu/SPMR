# SPMR: Saddle-point minimal residual

export
    spmr_sc, spmr_ns,
    spqmr_sc, spqmr_ns,
    recover_y

include("spmr_macros.jl")

@enum SpmrFlag begin
    CONVERGED
    MAXIT_EXCEEDED
    OTHER
end

@doc begin
"""
Status flag for SPMR solvers.

# Flags
-   `CONVERGED`:
    the relative residual or estimate thereof fell below the prescribed
    tolerance.
-   `MAXIT_EXCEEDED`: the maximum number of iterations was performed.
-   `OTHER`: some computed quantity became too small.
"""
end SpmrFlag

"""
Result struct for SPMR-SC and SPQMR-SC.

# Fields
-   `x`: ``\\vec{x}``.
-   `y`: ``\\vec{y}``.
-   `flag`: an [`SpmrFlag`](@ref) describing the terminal state of the iteration.
-   `iter`: the number of iterations performed.
-   `resvec`: the vector of relative residuals or estimates thereof.
"""
struct SpmrScResult
    x::Vector{Float64}
    y::Vector{Float64}
    flag::SpmrFlag
    iter::Int
    resvec::Vector{Float64}
end

function SpmrScResult(x::Vector{Float64}, y::Vector{Float64}, flag::SpmrFlag, iter::Int,
                      resvec::Vector{Float64}, precond::RealInvOperator)
    @ldiv_into!(ŷ, precond, y, length(y))
    return SpmrScResult(x, ŷ, flag, iter, resvec)
end

"""
Result struct for SPQMR-NS and SPQMR-NS.

# Fields
-   `x`: ``\\vec{x}``.
-   `flag`: an [`SpmrFlag`](@ref) describing the terminal state of the iteration.
-   `iter`: the number of iterations performed.
-   `resvec`: the vector of relative residuals or estimates thereof.

!!! note

    ``\\vec{y}`` must be recovered separately.
"""
struct SpmrNsResult
    x::Vector{Float64}
    flag::SpmrFlag
    iter::Int
    resvec::Vector{Float64}
end

const function_pairs = (:spmr_sc => :simba_sc, :spmr_ns => :simba_ns,
                        :spqmr_sc => :simbo_sc, :spqmr_ns => :simbo_ns)

const iteration_quotes = Dict(:spmr_sc =>
                              Dict(:init => :(),
                                   :compute_res => quote
                                       resvec[k] = k == 1 ? Ω.s : resvec[k-1] * Ω.s
                                       if resvec[k] < tol
                                           return SpmrScResult(x, y, CONVERGED,
                                                               k, resvec[1:k], precond)
                                       end
                                   end
                                  ),

                              :spmr_ns =>
                              Dict(:init => :(),
                                   :compute_res => quote
                                       resvec[k] = k == 1 ? Ω.s : resvec[k-1] * Ω.s
                                       if resvec[k] < tol
                                           return SpmrNsResult(-p, CONVERGED,
                                                               k, resvec[1:k])
                                       end
                                   end
                                  ),

                              :spqmr_sc =>
                              Dict(:init => quote
                                       relres = one(Float64)
                                       norm_g = BLAS.nrm2(m, g, 1)
                                   end,
                                   :compute_res => quote
                                       relres *= Ω.s
                                       resvec[k] = sqrt(k) * relres

                                       if resvec[k] < tol
                                           @mul_into!(r, K.G₂, x, m)
                                           r .-= g

                                           norm_r = BLAS.nrm2(m, r, 1)
                                           if norm_r < tol * norm_g
                                               return SpmrScResult(x, y, CONVERGED,
                                                                   k, resvec[1:k], precond)
                                           end
                                       end
                                   end
                                  ),

                              :spqmr_ns =>
                              Dict(:init => quote
                                       ℓ = nullsp_basis_size(K)

                                       relres = one(Float64)
                                       norm_f = BLAS.nrm2(n, f, 1)
                                   end,
                                   :compute_res => quote
                                       relres *= Ω.s
                                       resvec[k] = sqrt(k) * relres

                                       if resvec[k] < tol
                                           @mul_into!(p̂, K.A, p, n)
                                           @mul_into!(r, K.H₁', p̂, ℓ)
                                           r .-= fₕ

                                           norm_r = BLAS.nrm2(ℓ, r, 1)
                                           if norm_r < tol * norm_f
                                               return SpmrNsResult(-p, CONVERGED,
                                                                   k, resvec[1:k])
                                           end
                                       end
                                   end
                                  )
                             )

for (func, bidiag_func) in function_pairs
    if func == :spmr_sc || func == :spqmr_sc
        @eval begin
            function $func(K::SpmrScMatrix, g::AbstractVector{<:Real};
                           tol::Float64=1e-6, maxit::Int=10, precond::RealInvOperator=I)
                n, m = block_sizes(K)

                SSI, SI₀ = $bidiag_func(K, g, g, precond=precond)

                @init_qr(SI₀)
                @init_x!(x, SI₀, n)
                @init_y!(y, SI₀, m)

                if abs(SI₀.ξ) < eps()
                    return SpmrScResult(x, y, OTHER, 0, Float64[], precond)
                end

                resvec = Vector{Float64}(undef, min(m, maxit))

                $(iteration_quotes[func][:init])

                for (k, SI) in enumerate(SSI)
                    if k > maxit
                        return SpmrScResult(x, y, MAXIT_EXCEEDED, maxit, resvec[1:maxit],
                                            precond)
                    elseif abs(SI.ξ) < eps()
                        return SpmrScResult(x, y, OTHER, k-1, resvec[1:k-1], precond)
                    end

                    @update_qr!(Ω, SI)
                    @update_x!(x, SI, n)
                    @update_y!(y, SI, m)

                    $(iteration_quotes[func][:compute_res])
                end

                return SpmrScResult(x, y, MAXIT_EXCEEDED, m, resvec, precond)
            end
        end
    elseif func == :spmr_ns || func == :spqmr_ns
        @eval begin
            function $func(K::SpmrNsMatrix, f::AbstractVector{<:Real};
                           tol::Float64=1e-6, maxit::Int=10, precond::RealInvOperator=I)
                n, m = block_sizes(K)

                fₕ = K.H₁' * -f
                SNI, SI₀ = $bidiag_func(K, fₕ, fₕ, precond=precond)

                @init_qr(SI₀)
                @init_x!(p, SI₀, n)

                if abs(SI₀.ξ) < eps()
                    return SpmrNsResult(-p, OTHER, 0, Float64[])
                end

                resvec = Vector{Float64}(undef, min(n-m, maxit))

                $(iteration_quotes[func][:init])

                for (k, SI) in enumerate(SNI)
                    if k > maxit
                        return SpmrNsResult(-p, MAXIT_EXCEEDED, maxit, resvec[1:maxit])
                    elseif abs(SI.ξ) < eps()
                        return SpmrNsResult(-p, OTHER, k-1, resvec[1:k-1])
                    end

                    @update_qr!(Ω, SI)
                    @update_x!(p, SI, n)

                    $(iteration_quotes[func][:compute_res])
                end

                return SpmrNsResult(-p, MAXIT_EXCEEDED, n-m, resvec)
            end
        end
    end
end

@doc begin
"""
    spmr_sc(K, g; <keyword arguments>)

Solve the saddle-point system
```math
\\begin{bmatrix} A & G_1^T \\\\ G_2 & 0 \\end{bmatrix}
\\begin{bmatrix} \\vec{x} \\\\ \\vec{y} \\end{bmatrix} =
\\begin{bmatrix} \\vec{0} \\\\ \\vec{g} \\end{bmatrix}
```
by residual minimization using the Schur complement of ``A``,
where
```math
K =
\\begin{bmatrix} A & G_1^T \\\\ G_2 & 0 \\end{bmatrix}
```
is an [`SpmrScMatrix`](@ref).

The solution along with additional information is returned as an
[`SpmrScResult`](@ref).

# Arguments
- `tol::Float64=1e-6`: the relative residual tolerance.
- `maxit::Int=10`: the maximum number of iterations to perform.
- `precond::RealInvOperator=I`: a symmetric positive-definite preconditioner.
"""
end spmr_sc

@doc begin
"""
    spqmr_sc(K, g; <keyword arguments>)

Solve the saddle-point system
```math
\\begin{bmatrix} A & G_1^T \\\\ G_2 & 0 \\end{bmatrix}
\\begin{bmatrix} \\vec{x} \\\\ \\vec{y} \\end{bmatrix} =
\\begin{bmatrix} \\vec{0} \\\\ \\vec{g} \\end{bmatrix}
```
by residual quazi-minimization using the Schur complement of ``A``,
where
```math
K =
\\begin{bmatrix} A & G_1^T \\\\ G_2 & 0 \\end{bmatrix}
```
is an [`SpmrScMatrix`](@ref).

The solution along with additional information is returned as an
[`SpmrScResult`](@ref).

# Arguments
- `tol::Float64=1e-6`: the relative residual tolerance.
- `maxit::Int=10`: the maximum number of iterations to perform.
- `precond::RealInvOperator=I`: a symmetric positive-definite preconditioner.
"""
end spqmr_sc

@doc begin
"""
    spmr_ns(K, f; <keyword arguments>)

Solve the saddle-point system
```math
\\begin{bmatrix} A & G_1^T \\\\ G_2 & 0 \\end{bmatrix}
\\begin{bmatrix} \\vec{x} \\\\ \\vec{y} \\end{bmatrix} =
\\begin{bmatrix} \\vec{f} \\\\ \\vec{0} \\end{bmatrix}
```
by residual minimization using the nullspaces of ``G_1^T`` and ``G_2``,
where
```math
K =
\\begin{bmatrix} A & G_1^T \\\\ G_2 & 0 \\end{bmatrix}
```
is an [`SpmrNsMatrix`](@ref).

The solution along with additional information is returned as an
[`SpmrNsResult`](@ref).

# Arguments
- `tol::Float64=1e-6`: the relative residual tolerance.
- `maxit::Int=10`: the maximum number of iterations to perform.
- `precond::RealInvOperator=I`: a symmetric positive-definite preconditioner.
"""
end spmr_ns

@doc begin
"""
    spqmr_ns(K, f; <keyword arguments>)

Solve the saddle-point system
```math
\\begin{bmatrix} A & G_1^T \\\\ G_2 & 0 \\end{bmatrix}
\\begin{bmatrix} \\vec{x} \\\\ \\vec{y} \\end{bmatrix} =
\\begin{bmatrix} \\vec{f} \\\\ \\vec{0} \\end{bmatrix}
```
by residual quazi-minimization using the nullspaces of ``G_1^T`` and ``G_2``,
where
```math
K =
\\begin{bmatrix} A & G_1^T \\\\ G_2 & 0 \\end{bmatrix}
```
is an [`SpmrNsMatrix`](@ref).

The solution along with additional information is returned as an
[`SpmrNsResult`](@ref).

# Arguments
- `tol::Float64=1e-6`: the relative residual tolerance.
- `maxit::Int=10`: the maximum number of iterations to perform.
- `precond::RealInvOperator=I`: a symmetric positive-definite preconditioner.
"""
end spqmr_ns

function recover_y(result::SpmrNsResult, K::SpmrNsMatrix, G₁ᵀ::AbstractMatrix{<:Real},
                   f::Vector{<:Real})
    @mul_into!(x̂, K.A, result.x, block_sizes(K)[1])
    return G₁ᵀ \ (f - x̂)
end
