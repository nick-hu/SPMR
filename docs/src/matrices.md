# Saddle-point Matrices

```@meta
CurrentModule = SPMR
```

## Matrices for Schur Complement-Based Solvers

```@docs
SpmrScMatrix
SpmrScMatrix(::T, ::V, ::W) where {T<:InvLinearMap{Float64}, V<:RealOperator, W<:RealOperator}
SpmrScMatrix(::T, ::V, ::W) where {T<:AbstractMatrix{<:Real}, V<:RealOperator, W<:RealOperator}
block_sizes(::SpmrScMatrix)
```

## Matrices for Nullspace-Based Solvers

```@docs
SpmrNsMatrix
SpmrNsMatrix(::T, ::U, ::V, ::Integer) where {T<:RealOperator, U<:RealOperator, V<:RealOperator}
block_sizes(::SpmrNsMatrix)
nullsp_basis_size
```

## List of Matrix Types

```@docs
RealOperator
RealInvOperator
InvLinearMap
InvLinearMap(::Function, ::Int; kwargs...)
InvLinearMap(::Function, ::Function, ::Int; kwargs...)
```
