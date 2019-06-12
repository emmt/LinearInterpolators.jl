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

__precompile__(true)

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
    QuadraticSpline,
    QuadraticSplinePrime,
    RectangularSpline,
    RectangularSplinePrime,
    SafeFlat,
    SparseInterpolator,
    SparseUnidimensionalInterpolator,
    TabulatedInterpolator,
    boundaries,
    intercept,
    iscardinal,
    isnormalized,
    jacobian,
    nameof,
    rotate

include("AffineTransforms.jl")
using .AffineTransforms

include("kernels.jl")
using .Kernels

include("interpolations.jl")
using .Interpolations

end # module
