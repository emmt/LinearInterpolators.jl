#
# interpolations.jl --
#
# Implement linear interpolation.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2017, Éric Thiébaut.
# This file is part of TiPi.  All rights reserved.
#

module Interpolations

using Compat

import Base: eltype, length, size, first, last, clamp, convert

importall TiPi.Kernels
import TiPi.AffineTransforms: AffineTransform2D

import TiPi.Algebra: LinearOperator, apply_direct, apply_direct!,
    apply_adjoint, apply_adjoint!, output_size, input_size

export
    SparseInterpolator,
    limits,
    boundaries,
    inferior,
    superior,
    limits,
    computecoefs

"""
`support(A)` yields the size of the support of the interpolation kernel.
Argument can be an interpolation kernel, an interpolation method or an
interpolator.
"""
support{T,S,B}(::Kernel{T,S,B}) = S

"""
All interpolation limits inherit from the abstract type `Limits` and are the
combination of an extrapolation method and the length the dimension to
interpolate.
"""
@compat abstract type Limits{T<:AbstractFloat} end # FIXME: add parameter S

eltype{T}(::Limits{T}) = T
length{T}(B::Limits{T}) = B.len
size{T}(B::Limits{T}) = (B.len,)
size{T}(B::Limits{T}, i::Integer) =
    (i == 1 ? B.len : i > 1 ? 1 : throw(BoundsError()))
first{T}(B::Limits{T}) = 1
last{T}(B::Limits{T}) = B.len
clamp{T}(i, B::Limits{T}) = clamp(i, first(B), last(B))


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
# Direct linear interpolation

    apply_direct(ker, x, A)

interpolates array `A` with kernel `ker` at positions `x`, the result is an
array of same dimensions as `x`.

"""
function apply_direct end

"""

    apply_direct!(dst, K, x, A) -> dst

interpolates array `A` with kernel `K` at positions `x` and stores the result
in `dst` which is returned.  The destination array `dst` must have the same
size as `x`.

"""
function apply_direct! end

"""
# Adjoint of the linear interpolation


    apply_adjoint(ker, x, A, len)

apply the adjoint of the interpolation by kernel `ker` at positions `x` to the
array `A`.  Argument `A` must have the same dimensions as `x` and the result is
a vector of length `len`.

"""
function apply_adjoint end

"""
# The in-place version of the adjoint of the linear interpolation

## Unidimensional interpolation

    apply_adjoint!(dst, ker, x, A; clr=true) -> dst

apply the adjoint of the interpolation by kernel `ker` at positions `x` to the
array `A` and stores the result in `dst` which is returned.  Argument `A` must
have the same dimensions as `x`.  If `clr` is true, `dst` is zero-filled prior
to applying the operation, otherwise the content of `dst` is augmented by the
result of the operation.


## Separable multi-dimensional interpolation

    apply_adjoint!(dst, ker,        R, src, clr=true) -> dst
    apply_adjoint!(dst, ker1, ker2, R, src, clr=true) -> dst

applies the adjoint of the interpolation by and stores the result in `dst`
which is returned.  See `interp_direct!` for the arguments.

"""
function apply_adjoint! end

include("interp/flat.jl")
include("interp/safeflat.jl")
include("interp/sparse.jl")
include("interp/unidimensional.jl")
include("interp/separable.jl")
include("interp/nonseparable.jl")

end
