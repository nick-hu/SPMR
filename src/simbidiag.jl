# SIMBA/SIMBO: Simultaneous bidiagonalization via A-conjugacy/biorthogonality

include("simbidiag_macros.jl")

struct SpmrScIterate
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64

    ξ::Float64

    u::Vector{Float64}
    v::Vector{Float64}
    w::Vector{Float64}
    z::Vector{Float64}
end

struct SpmrNsIterate
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64

    ξ::Float64

    u::Vector{Float64}
    v::Vector{Float64}
    w::Vector{Float64}
    z::Vector{Float64}

    û::Vector{Float64}
    ŵ::Vector{Float64}
end

const bidiag_types = (:simba_sc => (:SpmrScMatrix, :SimbaScIterator, :SpmrScIterate),
                      :simba_ns => (:SpmrNsMatrix, :SimbaNsIterator, :SpmrNsIterate),
                      :simbo_sc => (:SpmrScMatrix, :SimboScIterator, :SpmrScIterate),
                      :simbo_ns => (:SpmrNsMatrix, :SimboNsIterator, :SpmrNsIterate))

const func_quotes = Dict(:simba_sc =>
                         Dict(:init => quote
                                  @normalize!(β, v, b, m)
                                  @normalize!(δ, z, c, m)
                              end,
                              :iterate => quote
                                  @mul_into!(v, K.G₁ᵀ', SI_prev.w, m)
                                  BLAS.axpy!(-SI_prev.α, SI_prev.v, v)
                                  @mul_into!(z, K.G₂, SI_prev.u, m)
                                  BLAS.axpy!(-SI_prev.γ, SI_prev.z, z)

                                  @normalize!(β, v, m)
                                  @normalize!(δ, z, m)
                              end
                             ),

                         :simbo_sc =>
                         Dict(:init => quote
                                  χ = c ⋅ b

                                  δ = sqrt(abs(χ))
                                  v = b / δ
                                  β = flipsign(δ, χ)
                                  z = c / β
                              end,
                              :iterate => quote
                                  @mul_into!(v, K.G₂, SI_prev.u, m)
                                  BLAS.axpy!(-SI_prev.γ, SI_prev.v, v)
                                  @mul_into!(z, K.G₁ᵀ', SI_prev.w, m)
                                  BLAS.axpy!(-SI_prev.α, SI_prev.z, z)

                                  χ = z ⋅ v

                                  δ = sqrt(abs(χ))
                                  BLAS.scal!(m, inv(δ), v, 1)
                                  β = flipsign(δ, χ)
                                  BLAS.scal!(m, inv(β), z, 1)
                              end
                             ),

                         :simba_ns =>
                         Dict(:init => quote
                                  ℓ = nullsp_basis_size(K)

                                  @normalize!(β, v, b, ℓ)
                                  @normalize!(δ, z, c, ℓ)
                              end,
                              :iterate => quote
                                  @mul_into!(v, K.H₂', SI_prev.ŵ, ℓ)
                                  BLAS.axpy!(-SI_prev.α, SI_prev.v, v)
                                  @mul_into!(z, K.H₁', SI_prev.û, ℓ)
                                  BLAS.axpy!(-SI_prev.γ, SI_prev.z, z)

                                  @normalize!(β, v, ℓ)
                                  @normalize!(δ, z, ℓ)
                              end
                             ),

                         :simbo_ns =>
                         Dict(:init => quote
                                  χ = c ⋅ b

                                  δ = sqrt(abs(χ))
                                  v = b / δ
                                  β = flipsign(δ, χ)
                                  z = c / β
                              end,
                              :iterate => quote
                                  @mul_into!(v, K.H₁', SI_prev.û, ℓ)
                                  BLAS.axpy!(-SI_prev.γ, SI_prev.v, v)
                                  @mul_into!(z, K.H₂', SI_prev.ŵ, ℓ)
                                  BLAS.axpy!(-SI_prev.α, SI_prev.z, z)

                                  χ = z ⋅ v

                                  δ = sqrt(abs(χ))
                                  BLAS.scal!(ℓ, inv(δ), v, 1)
                                  β = flipsign(δ, χ)
                                  BLAS.scal!(ℓ, inv(β), z, 1)
                              end
                             )
                        )

for (func, (matrix_type, iterator_type, iterate_type)) in bidiag_types
    @eval begin
        mutable struct $iterator_type
            K::$matrix_type
            SI::$iterate_type

            SI₀::$iterate_type  # Remember SI₀ so that we can reiterate
        end

        function $func(K::$matrix_type,
                       b::AbstractVector{<:Real},
                       c::AbstractVector{<:Real})
            return $func(K,
                         convert(Vector{Float64}, b),
                         convert(Vector{Float64}, c))
        end

        Base.eltype(::Type{$iterator_type}) = $iterate_type
    end

    if matrix_type == :SpmrScMatrix
        @eval Base.length(SSI::$iterator_type) = block_sizes(SSI.K)[2]
    elseif matrix_type == :SpmrNsMatrix
        @eval Base.length(SNI::$iterator_type) = foldl(-, block_sizes(SNI.K))
    end

    if func == :simba_sc || func == :simbo_sc
        @eval begin
            function $func(K::$matrix_type, b::Vector{Float64}, c::Vector{Float64})
                n, m = block_sizes(K)

                $(func_quotes[func][:init])

                @mul_into!(û, K.G₁ᵀ, v, n)
                @mul_into!(ŵ, K.G₂', z, n)

                @ldiv_into!(u, K.A, û, n)
                @ldiv_into!(w, K.Aᵀ, ŵ, n)

                ξ = û ⋅ w
                α = γ = sqrt(abs(ξ))

                @scal_signinv!(u, α, ξ, n)
                @scal_signinv!(w, γ, ξ, n)

                SI = $iterate_type(α, β, γ, δ, ξ, u, v, w, z)

                return ($iterator_type(K, SI, SI), SI)
            end

            function Base.iterate(SSI::$iterator_type, k::Int=0)
                n, m = block_sizes(SSI.K)

                k == 0 && (SSI.SI = SSI.SI₀)
                k ≥ m && return nothing

                K, SI_prev = SSI.K, SSI.SI

                $(func_quotes[func][:iterate])

                @mul_into!(û, K.G₁ᵀ, v, n)
                @mul_into!(ŵ, K.G₂', z, n)

                @ldiv_into!(u, K.A, û, n)
                BLAS.axpy!(-flipsign(β, SI_prev.ξ), SI_prev.u, u)
                @ldiv_into!(w, K.Aᵀ, ŵ, n)
                BLAS.axpy!(-flipsign(δ, SI_prev.ξ), SI_prev.w, w)

                ξ = û ⋅ w
                α = γ = sqrt(abs(ξ))

                @scal_signinv!(u, α, ξ, n)
                @scal_signinv!(w, γ, ξ, n)

                SSI.SI = $iterate_type(α, β, γ, δ, ξ, u, v, w, z)

                return (SSI.SI, k+1)
            end
        end
    elseif func == :simba_ns || func == :simbo_ns
        @eval begin
            function $func(K::$matrix_type, b::Vector{Float64}, c::Vector{Float64})
                n, _ = block_sizes(K)

                $(func_quotes[func][:init])

                @mul_into!(u, K.H₂, v, n)
                @mul_into!(w, K.H₁, z, n)

                @mul_into!(û, K.A, u, n)
                @mul_into!(ŵ, K.A', w, n)

                ξ = û ⋅ w
                α = γ = sqrt(abs(ξ))

                @scal_signinv!(u, α, ξ, n)
                @scal_signinv!(w, γ, ξ, n)

                SI = $iterate_type(α, β, γ, δ, ξ, u, v, w, z, û, ŵ)

                return ($iterator_type(K, SI, SI), SI)
            end

            function Base.iterate(SNI::$iterator_type, k::Int=0)
                n, m = block_sizes(SNI.K)
                ℓ = nullsp_basis_size(SNI.K)

                k == 0 && (SNI.SI = SNI.SI₀)
                k ≥ n-m && return nothing

                K, SI_prev = SNI.K, SNI.SI

                @scal_signinv!(SI_prev.û, SI_prev.α, SI_prev.ξ, n)
                @scal_signinv!(SI_prev.ŵ, SI_prev.γ, SI_prev.ξ, n)

                $(func_quotes[func][:iterate])

                @mul_into!(u, K.H₂, v, n)
                BLAS.axpy!(-flipsign(β, SI_prev.ξ), SI_prev.u, u)
                @mul_into!(w, K.H₁, z, n)
                BLAS.axpy!(-flipsign(δ, SI_prev.ξ), SI_prev.w, w)

                @mul_into!(û, K.A, u, n)
                @mul_into!(ŵ, K.A', w, n)

                ξ = û ⋅ w
                α = γ = sqrt(abs(ξ))

                @scal_signinv!(u, α, ξ, n)
                @scal_signinv!(w, γ, ξ, n)

                SNI.SI = $iterate_type(α, β, γ, δ, ξ, u, v, w, z, û, ŵ)

                return (SNI.SI, k+1)
            end
        end
    end
end
