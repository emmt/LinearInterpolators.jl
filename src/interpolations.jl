#
# interpolations.jl --
#
# Implement linear interpolation (here "linear" means that the result depends
# linearly on the interpolated data).
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2021, Éric Thiébaut.
#

module Interpolations

export
    SparseInterpolator,
    SparseUnidimensionalInterpolator,
    TabulatedInterpolator,
    TwoDimensionalTransformInterpolator,
    limits,
    inferior,
    superior,
    getcoefs

import Base: eltype, length, size, first, last, clamp, convert

using InterpolationKernels
import InterpolationKernels: boundaries

using TwoDimensional.AffineTransforms
using TwoDimensional: AffineTransform2D

using LazyAlgebra
using LazyAlgebra.Foundations
import LazyAlgebra: apply, apply!, vcreate, output_size, input_size,
    coefficients

"""

All interpolation limits inherit from the abstract type `Limits{T}` where `T`
is the floating-point type.  Interpolation limits are the combination of an
extrapolation method and the length of the dimension to interpolate.

"""
abstract type Limits{T<:AbstractFloat} end # FIXME: add parameter S

eltype(B::Limits) = eltype(typeof(B))
eltype(::Type{<:Limits{T}}) where {T} = T
length(B::Limits) = B.len
size(B::Limits) = (B.len,)
size(B::Limits, i::Integer) =
    (i == 1 ? B.len : i > 1 ? 1 : throw(BoundsError()))
first(B::Limits) = 1
last(B::Limits) = B.len
clamp(i, B::Limits) = clamp(i, first(B), last(B))

"""
    limits(ker::Kernel, len)

yields the concrete type descendant of `Limits` for interpolation with kernel
`ker` along a dimension of length `len` and applying the boundary conditions
embedded in `ker`.

"""
function limits end

"""
    getcoefs(ker, lim, x) -> j1, j2, ..., w1, w2, ...

yields the indexes of the neighbors and the corresponding interpolation weights
for interpolating at position `x` by kernel `ker` with the limits implemented
by `lim`.

"""
function getcoefs end

"""
    apply([P=Direct,] ker, x, src) -> dst

interpolates source array `src` at positions `x` with interpolation kernel
`ker` and yiedls the result as `dst`.

Interpolation is equivalent to applying a linear mapping.  Optional argument
`P` can be `Direct` or `Adjoint` to respectively compute the interpolation or
to apply the adjoint of the linear mapping implementing the interpolation.

"""
apply

"""
    apply!(dst, [P=Direct,] ker, x, src) -> dst

overwrites `dst` with the result of interpolating the source array `src` at
positions `x` with interpolation kernel `ker`.

Optional argument `P` can be `Direct` or `Adjoint`, see [`apply`](@ref) for
details.

"""
apply!

function rows end
function columns end
function fit end
function regularize end
function regularize! end
function inferior end
function superior end

include("interp/meta.jl")
import .Meta
include("interp/flat.jl")
include("interp/safeflat.jl")
include("interp/tabulated.jl")
using .TabulatedInterpolators
include("interp/sparse.jl")
using .SparseInterpolators
include("interp/unidimensional.jl")
using .UnidimensionalInterpolators
include("interp/separable.jl")
include("interp/nonseparable.jl")

end
