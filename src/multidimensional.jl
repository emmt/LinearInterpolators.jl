#
# multidimensional.jl -
#
# Implement interpolators able to operate on several dimensions at the same
# time.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2022, Éric Thiébaut.
#
module Multidimensional

export LazyMultidimInterpolator, SparseMultidimInterpolator

using ..LinearInterpolators
using ..LinearInterpolators: to_ntuple, to_unit_range, compute_indices,
    check_axes, check_indices, nth, argument_error, dimension_mismatch,
    expr_inline, expr_tuple

using InterpolationKernels
using InterpolationKernels: compute_offset_and_weights

using ZippedArrays

using LazyAlgebra
using LazyAlgebra: promote_multiplier

abstract type AbstractMultidimInterpolator{T,M,N,S} <: AbstractInterpolator{T,M,N} end

"""
    SparseMultidimInterpolator(pos, ker, cols, bnd=Flat())

yields a multi-dimensional interpolator with pre-computed coefficients.
Arguments are:

- `pos` is an `M`-dimensional array specifying the coordinates interpolating
  positions.  The size of `pos` is that of the result of the interpolation.
  The elements of `pos` are `N`-tuples of fractional coordinates in the arrays
  to be interpolated.  Interpolating positions may also be specified as a
  `N`-tuples of arrays of coordinates.  If `N = 1` (i.e., interpolation of
  unidimensional arrays), `pos` may just be an array of coordinates.

- `ker` specifies the interpolation kernel to use along all dimensions or a
  `N`-tuple of interpolation kernels to have a, possibly, different
  interpolation kernel along each dimension of interpolation.

- `cols` specifies the `N`-dimensional size of the arrays to be interpolated.

- Optional argument `bnd` can be used to specify different boundary conditions
  than the default ones.  This optional argument can be a single instance or a
  `N`-tuple of boundary conditions depending whether the same or different
  boundary conditions are assumed for the `N` dimensions of interpolation.

A sparse interpolator is meant to interpolate `N`-dimensional arrays so as to
produce `M`-dimensional arrays.  The interpolation is done separately along the
`N` dimensions of the interpolated array.

The fractional coordinates `pos` can also be specified by an affine transform,
say `xform`, (from `M`-dimensional coordinates to `N`-dimensional coordinates)
and the `M`-tuple `rows` of the dimensions of the interpolation result:

    SparseMultidimInterpolator(xform, rows, ker, cols, bnd)

is equivalent to take:

    pos[i] = xform.(CartesianIndices(rows)[i])

"""
struct SparseMultidimInterpolator{T, # type of interpolation weights
                                  M, # number of output dimensions
                                  N, # number of input dimensions
                                  S, # N-tuple of kernel lengths
                                  W, # type of weights
                                  I, # type of indices
                                  } <: AbstractMultidimInterpolator{T,M,N,S}
    rows::Dims{M}
    cols::Dims{N}
    wgt::W # N-tuple of arrays of interpolation weights
    ind::I # N-tuple of arrays of interpolation weights
    function SparseMultidimInterpolator{T,M,N,S}(
        rows::NTuple{M,Integer},
        cols::NTuple{N,Integer},
        wgt::W,
        ind::I;
        inbounds::Bool=false) where {T,M,N,S,
                                     # FIXME: In the future, code can be
                                     # generalized to abstract arrays but the
                                     # indexing style must then be determined.
                                     W<:NTuple{N,Array{<:Tuple{Vararg{T}}}},
                                     I<:NTuple{N,Array{<:Tuple{Vararg{Int}}}}}
        S isa NTuple{N,Int} || error("invalid type parameter S")
        for d in 1:M
            rows[d] > 0 || argument_error(
                "invalid $(nth(d)) output dimension")
        end
        ncols = 1
        for d in 1:N
            cols[d] > 0 || argument_error(
                "invalid $(nth(d)) input dimension")
            ncols *= Int(cols[d])
            eltype(wgt[d]) == NTuple{S[d],T} || error(
                "element type of $(nth(d)) array of weights must be NTuple{$(S[d]),$T}")
            size(wgt[d]) == rows || dimension_mismatch(
                "$(nth(d)) array weights has invalid size")
            eltype(ind[d]) == NTuple{S[d],Int} || error(
                "element type of $(nth(d)) array of indices must be NTuple{$(S[d]),Int}")
            size(ind[d]) == rows || dimension_mismatch(
                "$(nth(d)) array of indices has invalid size")
            inbounds || check_indices(ind[d], Base.OneTo(ncols)) || error(
                "out of bound index in $(nth(d)) array of indices")
        end
        return new{T,M,N,S,W,I}(rows, cols, wgt, ind)
    end
end

"""
    LazyMultidimInterpolator{T,M,N,S}(...)

yields a separable interpolator whose coefficients are computed on the fly.
Type parameters are: `T` the floating-point type of the coefficients, `M` the
number of dimensions of the output, `N` the number of dimensions of the input,
and `S` the `N`-tuple of interpolation kernels lengths.

Such an operator can be used to interpolate a `N`-dimensional array in order to
produce a `M`-dimensional result.

"""
struct LazyMultidimInterpolator{T, # type of interpolation weights
                                M, # number of output dimensions
                                N, # number of input dimensions
                                S, # N-tuple of kernel lengths
                                P, # type of interpolation coordinates
                                K, # type of kernels
                                B, # type of boundary conditions
                                } <: AbstractMultidimInterpolator{T,M,N,S}
    rows::Dims{M}
    cols::Dims{N}
    pos::P # N-tuple of arrays of interpolated coordinates
    ker::K # N-tuple of interpolation kernels
    bnd::B # N-tuple of boundary conditions
    function LazyMultidimInterpolator{T,M,N,S}(
        rows::NTuple{M,Integer},
        cols::NTuple{N,Integer},
        pos::P,
        ker::K,
        bnd::B) where {T,M,N,S,
                       # FIXME: In the future, code can be generalized to
                       # abstract arrays but the indexing style must then be
                       # determined.
                       P<:AbstractArray{NTuple{N,T},M},
                       K<:NTuple{N,Kernel{T}},
                       B<:NTuple{N,BoundaryConditions}}
        S isa NTuple{N,Int} || error("invalid type parameter S")
        for d in 1:M
            rows[d] > 0 || argument_error(
                "invalid $(nth(d)) output dimension")
        end
        size(pos) == rows || dimension_mismatch(
            "array of interpolated positions has invalid size")
        Base.has_offset_axes(pos) && argument_error(
            "array of interpolated positions has non-standard indices")
        for d in 1:N
            cols[d] > 0 || argument_error(
                "invalid $(nth(d)) input dimension")
            length(ker[d]) == S[d] || error(
                "length of $(nth(d)) kernel must be $(S[d])")
        end
        return new{T,M,N,S,P,K,B}(rows, cols, pos, ker, bnd)
    end
end

cols(A::AbstractMultidimInterpolator) = getfield(A, :cols)
rows(A::AbstractMultidimInterpolator) = getfield(A, :rows)
LazyAlgebra.output_size(A::AbstractMultidimInterpolator) = rows(A)
LazyAlgebra.input_size(A::AbstractMultidimInterpolator) = cols(A)
LazyAlgebra.row_size(A::AbstractMultidimInterpolator) = rows(A)
LazyAlgebra.col_size(A::AbstractMultidimInterpolator) = cols(A)

function Base.show(io::IO, A::SparseMultidimInterpolator{T,M,N,S}) where {T,M,N,S}
    print(io, "SparseMultidimInterpolator{$T,$M,$N,$S}: ")
    join(io, cols(A), "×")
    print(io, " → ")
    join(io, rows(A), "×")
end

function Base.show(io::IO, A::LazyMultidimInterpolator{T,M,N,S}) where {T,M,N,S}
    print(io, "LazyMultidimInterpolator{$T,$M,$N,$S}: ")
    join(io, cols(A), "×")
    print(io, " → ")
    join(io, rows(A), "×")
end

const Repeatable{T,N} = Union{T,NTuple{N,T}}

# Common constructors.
for func in (:SparseMultidimInterpolator, :LazyMultidimInterpolator)
    @eval begin

        # Provide element type.
        function $func(pos::AbstractArray{<:Real,M},
                       ker::Repeatable{Kernel,1},
                       cols::Repeatable{Integer,1},
                       bnd::Repeatable{BoundaryConditions,1} = Flat(),
                       ) where {M}
            T = operator_eltype(pos, ker)
            return $func{T}(pos, ker, cols, bnd)
        end
        function $func(pos::Union{NTuple{N,AbstractArray{<:Real,M}},
                                  AbstractArray{<:NTuple{N,Real},M}},
                       ker::Repeatable{Kernel,N},
                       cols::NTuple{N,Integer},
                       bnd::Repeatable{BoundaryConditions,N}) where {M,N}
            T = operator_eltype(pos, ker)
            return $func{T}(pos, ker, cols, bnd)
        end

        # Convert arguments.
        function $func{T}(pos::AbstractArray{<:Real,M},
                          ker::Repeatable{Kernel,1},
                          cols::Repeatable{Integer,1},
                          bnd::Repeatable{BoundaryConditions,1} = Flat(),
                          ) where {T,M}
            return $func{T,M,1}(
                convert_coord($func{T,M,1}, pos),
                convert_kernel($func{T,M,1}, ker),
                Dims{1}(cols),
                to_ntuple(Tuple{BoundaryConditions}, bnd))
        end
        function $func{T}(pos::Union{NTuple{N,AbstractArray{<:Real,M}},
                                     AbstractArray{<:NTuple{N,Real},M}},
                          ker::Repeatable{Kernel,N},
                          cols::NTuple{N,Integer},
                          bnd::Repeatable{BoundaryConditions,N} = Flat(),
                          ) where {T,M,N}
            return $func{T,M,N}(
                convert_coord($func{T,M,N}, pos),
                convert_kernel($func{T,M,N}, ker),
                Dims{N}(cols),
                to_ntuple(NTuple{N,BoundaryConditions}, bnd))
        end

    end
end

# The following constructors are the ones which are eventually called with type
# parameters T, M, and N to avoid ambiguities.  They shall not be directly
# called.
function SparseMultidimInterpolator{T,M,N}(
    pos::AbstractArray{<:NTuple{N,Real},M},
    ker::NTuple{N,Kernel{T}},
    cols::Dims{N},
    bnd::NTuple{N,BoundaryConditions}) where {T,M,N}

    Base.has_offset_axes(pos) && error("only standard indices are supported")
    rows = size(pos)
    S = map(length, ker)
    wgt = ntuple(d -> Array{NTuple{S[d],T}}(undef, rows), Val(N))
    ind = ntuple(d -> Array{NTuple{S[d],Int}}(undef, rows), Val(N))
    for d in 1:N
        instantiate!(wgt[d], ind[d], with_eltype(T, ker[d]),
                     to_unit_range(cols[d]), pos, d, bnd[d])
    end
    return SparseMultidimInterpolator{T,M,N,S}(rows, cols, wgt, ind;
                                               inbounds=true)
end

function LazyMultidimInterpolator{T,M,N}(
    pos::AbstractArray{NTuple{N,T},M},
    ker::NTuple{N,Kernel{T}},
    cols::Dims{N},
    bnd::NTuple{N,BoundaryConditions}) where {T,M,N}

    Base.has_offset_axes(pos) && error("only standard indices are supported")
    rows = size(pos)
    S = map(length, ker)
    return LazyMultidimInterpolator{T,M,N,S}(rows, cols, pos, ker, bnd)
end

# Auxiliary function to determine element type of operator.
operator_eltype(pos, ker::Repeatable{Kernel}) =
    float(promote_type(promote_coord_eltype(pos),
                       promote_kernel_eltype(ker)))

# Promote element type for kernel(s) argument.
promote_kernel_eltype(ker::Kernel) = eltype(ker)
promote_kernel_eltype(ker::Tuple{Vararg{Kernel}}) =
    promote_eltype(map(eltype, ker)...)

# Promote element type for coordinates argument.
promote_coord_eltype(pos::AbstractArray) = eltype(pos)
promote_coord_eltype(pos::Tuple{Vararg{AbstractArray}}) =
    promote_eltype(map(eltype, pos)...)
promote_coord_eltype(pos::AbstractArray{<:NTuple}) =
    promote_eltype(fieldtypes(eltype(pos))...)

# Convert argument specifying interpolation coordinates in constructors.
function convert_coord(::Type{<:SparseMultidimInterpolator{T,M,1}},
                       pos::AbstractArray{<:Real,M}) where {T,M}
    return ZippedArray(pos)
end
function convert_coord(::Type{<:SparseMultidimInterpolator{T,M,N}},
                       pos::NTuple{N,AbstractArray{<:Real,M}}) where {T,M,N}
    return ZippedArray(pos...)
end
function convert_coord(::Type{<:SparseMultidimInterpolator{T,M,N}},
                       pos::AbstractArray{<:NTuple{N,Real},M}) where {T,M,N}
    return pos
end

# For lazy operators the element type of arguments is converted to avoid
# repeated conversions.  FIXME: avoid using a zipped array if conversion is
# needed?
function convert_coord(::Type{<:LazyMultidimInterpolator{T,M,1}},
                       pos::AbstractArray{<:Real,M}) where {T,M}
    return ZippedArray(with_eltype(T, pos))
end
function convert_coord(::Type{<:LazyMultidimInterpolator{T,M,N}},
                       pos::NTuple{N,AbstractArray{<:Real,M}}) where {T,M,N}
    return ZippedArray(map(A -> with_eltype(T, A), pos)...)
end
function convert_coord(::Type{<:LazyMultidimInterpolator{T,M,N}},
                       pos::AbstractArray{<:NTuple{N,Real},M}) where {T,M,N}
    return with_eltype(NTuple{N,T}, pos)
end
function convert_coord(::Type{<:LazyMultidimInterpolator{T,M,N}},
                       pos::AbstractArray{<:NTuple{N,T},M}) where {T,M,N}
    return pos
end

# Convert argument specifying interpolation kernel(s) in constructors.
function convert_kernel(A::Type{<:AbstractMultidimInterpolator{T,M,N}},
                        ker::Kernel) where {T,M,N}
    ker_d = with_eltype(T, ker)
    return ntuple(d -> ker_d, Val(N))
end
function convert_kernel(A::Type{<:AbstractMultidimInterpolator{T,M,N}},
                        ker::NTuple{N,Kernel}) where {T,M,N}
    return ntuple(d -> with_eltype(T, ker[d]), Val(N))
end
function convert_kernel(A::Type{<:AbstractMultidimInterpolator{T,M,N}},
                        ker::NTuple{N,Kernel{T}}) where {T,M,N}
    return ker
end

# Build a sparse interpolator from a lazy version.
SparseMultidimInterpolator(A::LazyMultidimInterpolator) =
    SparseMultidimInterpolator(A.pos, A.ker, cols(A), A.bnd)
SparseMultidimInterpolator{T}(A::LazyMultidimInterpolator) where {T} =
    SparseMultidimInterpolator{T}(A.pos, A.ker, cols(A), A.bnd)

function instantiate!(wgt::AbstractArray{NTuple{S,T},M},
                      ind::AbstractArray{NTuple{S,Int},M},
                      ker::Kernel{T,S},
                      rng::AbstractUnitRange{Int},
                      pos::AbstractArray{<:NTuple{N,Real},M},
                      d::Int,
                      bnd::BoundaryConditions) where {T,M,N,S}
    1 ≤ d ≤ N || error("out of bounds dimension index")
    for i in eachindex(pos)
        x = convert(T, pos[i][d])
        t, w = compute_offset_and_weights(ker, x)
        wgt[i] = w
        ind[i] = compute_indices(bnd, t, rng, Val(S))
    end
end

function LazyAlgebra.vcreate(::Type{LazyAlgebra.Direct},
                             A::AbstractMultidimInterpolator{<:Any,M,N},
                             x::AbstractArray{<:Any,N},
                             scratch::Bool) where {M,N}
    T = promote_type(eltype(A), eltype(x))
    return Array{T,M}(undef, rows(A))
end

function LazyAlgebra.vcreate(::Type{LazyAlgebra.Adjoint},
                             A::AbstractMultidimInterpolator{<:Any,M,N},
                             x::AbstractArray{<:Any,M},
                             scratch::Bool) where {M,N}
    T = promote_type(eltype(A), eltype(x))
    return Array{T,N}(undef, cols(A))
end

function LazyAlgebra.apply!(alpha::Number,
                            ::Type{LazyAlgebra.Direct},
                            A::AbstractMultidimInterpolator{<:Any,M,N},
                            x::AbstractArray{<:Any,N},
                            scratch::Bool,
                            beta::Number,
                            y::AbstractArray{<:Any,M}) where {M,N}
    check_axes(x, cols(A)) || dimension_mismatch(
        "input array has incompatible indices")
    check_axes(y, rows(A)) || dimension_mismatch(
        "output array has incompatible indices")
    if alpha == zero(alpha)
        vscale!(y, beta)
    elseif alpha == oneunit(alpha)
        if beta == zero(beta)
            unsafe_apply_direct!(
                LazyAlgebra.axpby_yields_x,     1, A, x, 0, y)
        elseif beta == oneunit(beta)
            unsafe_apply_direct!(
                LazyAlgebra.axpby_yields_xpy,   1, A, x, 1, y)
        else
            β = promote_multiplier(beta, promote_eltype(A, x))
            unsafe_apply_direct!(
                LazyAlgebra.axpby_yields_xpby,  1, A, x, β, y)
         end
    else
        α = promote_multiplier(alpha, promote_eltype(A, x))
        if beta == zero(beta)
            unsafe_apply_direct!(
                LazyAlgebra.axpby_yields_ax,    α, A, x, 0, y)
        elseif beta == oneunit(beta)
            unsafe_apply_direct!(
                LazyAlgebra.axpby_yields_axpy,  α, A, x, 1, y)
        else
            β = promote_multiplier(beta, promote_eltype(A, x))
            unsafe_apply_direct!(
                LazyAlgebra.axpby_yields_axpby, α, A, x, β, y)
        end
    end
    return y
end

@generated function unsafe_apply_direct!(
    f::Function,
    α::Number,
    A::T,
    x::AbstractArray{<:Any,N},
    β::Number,
    y::AbstractArray{<:Any,M}) where {
        M,N,S,T<:AbstractMultidimInterpolator{<:Any,M,N,S}}
    return unsafe_apply_direct!(
        Expr, collect(S), Val(T <: LazyMultidimInterpolator))
end

function LazyAlgebra.apply!(alpha::Number,
                            ::Type{LazyAlgebra.Adjoint},
                            A::AbstractMultidimInterpolator{<:Any,M,N},
                            x::AbstractArray{<:Any,M},
                            scratch::Bool,
                            beta::Number,
                            y::AbstractArray{<:Any,N}) where {M,N}
    check_axes(x, rows(A)) || dimension_mismatch(
        "input array has incompatible indices")
    check_axes(y, cols(A)) || dimension_mismatch(
        "output array has incompatible indices")
    vscale!(y, beta)
    if alpha != zero(alpha)
        α = promote_multiplier(alpha, promote_eltype(A, x))
        unsafe_apply_adjoint!(y, α, A, x)
    end
    return y
end

@generated function unsafe_apply_adjoint!(
    y::AbstractArray{<:Any,M},
    α::Number,
    A::T,
    x::AbstractArray{<:Any,N}) where {
        M,N,S,T<:AbstractMultidimInterpolator{<:Any,M,N,S}}
    return unsafe_apply_adjoint!(
        Expr, collect(S), Val(T <: LazyMultidimInterpolator))
end

# Auxiliary functions to generate symbols.
j_(d::Int) = Symbol(:j_,d)
j_(d::Int, k::Int) = :($(j_(d))[$k])
k_(d::Int) = Symbol(:k_,d)
t_(d::Int) = Symbol(:t_,d)
w_(d::Int) = Symbol(:w_,d)
w_(d::Int, k::Int) = :($(w_(d))[$k])
x_(d::Int) = Symbol(:x_,d)
z_(d::Int) = Symbol(:z_,d)

# Generate code to extract (lazy=false) or compute (lazy=true) interpolation
# weights and indices at index i.
function start_block(S::Vector{Int}, lazy::Val{false})
    ndims = length(S)
    ex = Expr(:block)
    for d in 1:ndims
        push!(
            ex.args,
            # w_$d = A.wgt[d][i]
            :($(w_(d)) = A.wgt[$d][i]),
            # j_$d = A.ind[d][i]
            :($(j_(d)) = A.ind[$d][i]))
    end
    return ex
end

function start_block(S::Vector{Int}, lazy::Val{true})
    ndims = length(S)
    ex = Expr(:block)
    for d in 1:ndims
        push!(
            ex.args,
            # (t_$d, w_$d) = compute_offset_and_weights(A.ker[d], A.pos[i][d])
            :(($(t_(d)), $(w_(d))) = compute_offset_and_weights(
                A.ker[$d], A.pos[i][$d])),
            # j_$d = compute_indices(A.bnd[d], t_$d, col_size[d], Val(S[d])))
            :($(j_(d)) = compute_indices(
                A.bnd[$d], $(t_(d)), A.cols[$d], Val($(S[d])))))
    end
    return ex
end

"""
    unsafe_apply_direct!(::Type{Expr}, S, lazy)

yields the code of the `unsafe_apply_direct!` method.  Argument `S` is the
vector of kernel lengths along each dimension.  If `lazy` is `Val(true)`, code
for the lazy version of the interpolator is generated.

The following pseudo-code implements the factorized form of a 3-D interpolation
with pre-computed coefficients.  In the lazy version, only the first lines
(extraction of the interpolation weights and indices) change.

```julia
for i in eachindex(y)
    # Get interpolation weights and indices for i-th entry.
    w_1, w_2, ..., w_N = A.wgt[1][i], A.wgt[2][i], ..., A.ind[N][i]
    j_1, j_2, ..., j_N = A.ind[1][i], A.ind[2][i], ..., A.ind[N][i]
    # Compute interpolated value.
    z = (((x[j_1[1], j_2[1], j_3[1]]*w_1[1] +
           x[j_1[2], j_2[1], j_3[1]]*w_1[2] + ...)*w_2[1] +
          (x[j_1[1], j_2[2], j_3[1]]*w_1[1] +
           x[j_1[2], j_2[2], j_3[1]]*w_1[2] + ...)*w_2[2] + ...)*w_3[1] +
         ((x[j_1[1], j_2[1], j_3[2]]*w_1[1] +
           x[j_1[2], j_2[1], j_3[2]]*w_1[2] + ...)*w_2[1] +
          (x[j_1[1], j_2[2], j_3[2]]*w_1[1] +
           x[j_1[2], j_2[2], j_3[2]]*w_1[2] + ...)*w_2[2] + ...)*w_3[2] + ...)
    # Apply linear combination.
    y[i] = f(α, z, β, y[i])
end
```

"""
function unsafe_apply_direct!(::Type{Expr}, S::Vector{Int}, lazy::Val)
    # Extract weights and indices at index i.
    ex = start_block(S, lazy)

    # Auxiliary function to generate nested inner products.
    function inner(d::Int, S::Vector{Int}, j_tail)
        local ex = Expr(:call, :(+))
        for k ∈ 1:S[d]
            if d == 1 # First dimension.
                a = :(x[$(j_(d,k)), $(j_tail...)])
            else # Other dimensions.
                a = inner(d - 1, S, (j_(d,k), j_tail...))
            end
            push!(ex.args, :($a*$(w_(d,k))))
        end
        return ex
    end
    push!(ex.args,
          :(z = $(inner(length(S), S, ()))),
          :(y[i] = f(α, z, β, y[i])))
    return quote
        @inbounds for i in eachindex(y)
            $(ex.args...)
        end
        return nothing
    end
end

"""
    unsafe_apply_adjoint!(::Type{Expr}, S, lazy) -> ex

yields the code of the `unsafe_apply_adjoint!` method.  Argument `S` is the
vector of kernel lengths along each dimension.  If `lazy` is `Val(true)`, code
for the lazy version of the interpolator is generated.

The following pseudo-code implements the factorized form of the adjoint
4-D interpolation (after the destination y has been pre-multiplied by β).
This pseudo-code is written so as to make the recursive structure more
obvious.  In the generated function, at least the innermost loops on the
kernels sizes S are unrolled.  In the lazy version, only the first lines
(extraction of the interpolation weights and indices) change.

```julia
for i in eachindex(x)
    z = α*x[i]
    if z != zero(z)
        w_1, w_2, ..., w_N = A.wgt[1][i], A.wgt[2][i], ..., A.ind[N][i]
        j_1, j_2, ..., j_N = A.ind[1][i], A.ind[2][i], ..., A.ind[N][i]
        z_5 = z
        k_5 = ()
        for s4 in 1:S[4] # last dimension
            z_4 = w_4[s4]*z_5
            k_4 = (j_4[s4], k_5...)
            for s3 in 1:S[3]
                z_3 = w_3[s3]*z_4
                k_3 = (j_3[s3], k_4...)
                for s2 in 1:S[2]
                    z_2 = w_2[s2]*z_3
                    k_2 = (j_2[s2], k_3...)
                    for s1 in 1:S[1] # first dimension
                        y[j_1[s1],k_2...] += w_1[s1]*z_2
                    end
                end
            end
        end
    end
end
```

"""
function unsafe_apply_adjoint!(::Type{Expr}, S::Vector{Int}, lazy::Val)
    # Extract weights and indices at index i.
    ex = start_block(S, lazy)

    # Initialize variables needed by the recursion.
    ndims = length(S)
    let d = ndims+1
        push!(ex.args,
              # z_$d = z
              :($(z_(d)) = z),
              # k_$d = ()
              :($(k_(d)) = ()))
    end

    # Recurse over dimensions.
    function recurse!(ex::Expr, d::Int, S::Vector{Int})
        if d > 1
            # unroll loop for other dimensions than the first one
            for s in 1:S[d]
                push!(ex.args,
                      # z_$d = w_$d[s]*z_$(d+1)
                      :($(z_(d)) = $(w_(d,s))*$(z_(d+1))),
                      # k_$d = (j_$d[s], k_$(d+1)...)
                      :($(k_(d)) = ($(j_(d,s)), $(k_(d+1))...)))
                recurse!(ex, d - 1, S)
            end
        else
            # unroll loop for first dimension
            for s in 1:S[d]
                push!(ex.args, :(y[$(j_(d,s)), $(k_(d+1))...]
                                 += $(w_(d,s))*$(z_(d+1))))
            end
        end
        return ex
    end
    return quote
        #$(expr_inline())
        for i in eachindex(x)
            z = α*x[i]
            if z != zero(z)
                $(recurse!(ex, ndims, S).args...)
            end
        end
        return nothing
    end
end

end # module Multidimensional
