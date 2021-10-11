#
# LinearInterpolators.jl -
#
# Implement various interpolation methods as linear mappings.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2018, Éric Thiébaut.
#

module LinearInterpolators

# Export public types and methods.  Methods like `translate`, `compose`,
# `scale`, etc. which are accessible via operators like `+` or `*` or `∘` are
# however not exported.
export
    AffineTransform2D,
    Boundaries,
    CardinalCubicSpline,
    CardinalCubicSplinePrime,
    CatmullRomSpline,
    CatmullRomSplinePrime,
    CubicSpline,
    CubicSplinePrime,
    Flat,
    Kernel,
    KeysSpline,
    KeysSplinePrime,
    LanczosKernel,
    LanczosKernelPrime,
    LinearSpline,
    LinearSplinePrime,
    MitchellNetravaliSpline,
    MitchellNetravaliSplinePrime,
    QuadraticSpline,
    QuadraticSplinePrime,
    RectangularSpline,
    RectangularSplinePrime,
    SafeFlat,
    SparseInterpolator,
    SparseUnidimensionalInterpolator,
    TabulatedInterpolator,
    TwoDimensionalTransformInterpolator,
    boundaries,
    intercept,
    iscardinal,
    isnormalized,
    nameof,
    rotate

using InterpolationKernels
import InterpolationKernels: boundaries

using TwoDimensional.AffineTransforms
using TwoDimensional: AffineTransform2D

using LazyAlgebra
using LazyAlgebra.Foundations
import LazyAlgebra: apply, apply!, vcreate, output_size, input_size,
    coefficients

import Base: eltype, length, size, first, last, clamp, convert

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

include("types.jl")
include("meta.jl")
import .Meta
include("boundaries.jl")
include("tabulated.jl")
using .TabulatedInterpolators
include("sparse.jl")
using .SparseInterpolators
include("unidimensional.jl")
using .UnidimensionalInterpolators
include("separable.jl")
include("nonseparable.jl")

end # module
