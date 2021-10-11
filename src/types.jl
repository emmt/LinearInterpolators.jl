#
# types.jl -
#
# Definitions of common type in `LinearInterpolators`.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2021, Éric Thiébaut.
#

"""

All interpolator types inherit from `AbstractInterpolator{T}` which
is a linear mapping with coefficients of floating-point type `T`.

"""
abstract type AbstractInterpolator{T<:AbstractFloat} <: LinearMapping end

"""

All extrapolation methods (a.k.a. boundary conditions) are singletons and
inherit from the abstract type `Boundaries{T,S,R}`.  Instances of boundary
conditions must know the floating-point type `T` used for computations, the
finite size `S` of the interpolation kernel and the range of valid indices
along the interpolated dimension of type `R<:AbstractUnitRange{Int}`.  The
method `axis(B::Boundaries)` yields the range of valid indices along the
interpolated dimension.

See [`Flat`](@ref) and  [`SafeFlat`](@ref).

"""
abstract type Boundaries{T<:AbstractFloat,S,R<:AbstractUnitRange{Int}} end

"""
    Flat{T,S}(axis)

yields an instance of *flat* boundary conditions for interpolationg along a
dimension whose range of indices is specified by `axis` (the range of indices
or the length of the dimension), using floating-point type `T` for computations
and with a kernel of support size `S`.

*Flat* boundary conditions assume that the value of the nearest sample is used
when extrapolating.  Compared to [`SafeFlat`](@ref), this version assumes that
no integer overflows occur when converting coordinates to indices.

Parameters `T` and `S` may be avoided by specifying the interpolation kernel:

    Flat(ker::Kernel{T,S}, axis)

See [`Boundaries`](@ref) and  [`SafeFlat`](@ref).

"""
struct Flat{T<:AbstractFloat,S,R<:AbstractUnitRange{Int}} <: Boundaries{T,S,R}
    off_2::Int # first(axis) - 1
    off_3::Int # last(axis) - S
    axis::R    # range of indices in interpolated dimension
    function Flat{T,S,R}(axis::R) where {T<:AbstractFloat,S,
                                         R<:AbstractUnitRange{Int}}
        (isa(S, Int) && S ≥ 1) || bad_type_parameter(:S, S, Int)
        return new{T,S,R}(first(axis) - 1, last(axis) - S, axis)
    end
end

"""
    SafeFlat{T,S}(axis)

yields an instance of *safe flat* boundary conditions for interpolationg along
a dimension whose range of indices is specified by `axis` (the range of indices
or the length of the dimension), using floating-point type `T` for computations
and with a kernel of finite size `S`.

*Safe flat* boundary conditions assume that the value of the nearest sample is
used when extrapolating.  Compared to [`Flat`](@ref), this version account for
possible integer overflows when converting coordinates to indices.  As a
consequence, applying `SafeFlat` boundary conditions is always safe but slower
than applying [`Flat`](@ref) boundary conditions.

Parameters `T` and `S` may be avoided by specifying the interpolation kernel:

    SafeFlat(ker::kernel{T,S}, axis)

See [`Boundaries`](@ref) and  [`Flat`](@ref).

"""
struct SafeFlat{T<:AbstractFloat,S,R<:AbstractUnitRange{Int}} <: Boundaries{T,S,R}
    off_1::T # first(axis) - S
    off_2::T # first(axis) - 1
    off_3::T # last(axis) - S
    off_4::T # last(axis) - 1
    axis::R  # range of indices in interpolated dimension
    function SafeFlat{T,S,R}(axis::R) where {T<:AbstractFloat,S,
                                             R<:AbstractUnitRange{Int}}
        (isa(S, Int) && S ≥ 1) || bad_type_parameter(:S, S, Int)
        return new{T,S,R}(first(axis) - S, first(axis) - 1,
                          last(axis) - S, last(axis) - 1, axis)
    end
end

struct LazyInterpolator{B<:Boundaries,T,N,
                        K<:Kernel{T},
                        A<:AbstractArray{T,N}} <: AbstractInterpolator{T}
    ker::K
    pos::A
end

struct LazySeparableInterpolator{B<:Boundaries,D,T,
                                 K<:Kernel{T},
                                 A<:AbstractVector{T}} <: AbstractInterpolator{T}
    ker::K
    pos::A
end
