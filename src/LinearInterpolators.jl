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
    CatmullRomSpline,
    CubicSpline,
    Flat,
    Kernel,
    KeysSpline,
    LanczosKernel,
    LinearSpline,
    MitchellNetravaliSpline,
    QuadraticSpline,
    RectangularSpline,
    SafeFlat,
    SparseInterpolator,
    boundaries,
    intercept,
    iscardinal,
    isnormalized,
    jacobian,
    rotate

include("AffineTransforms.jl")
using .AffineTransforms

include("kernels.jl")
using .Kernels

include("interpolations.jl")
using .Interpolations

end # module
