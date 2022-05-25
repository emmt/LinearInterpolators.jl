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
    # Re-export symbols from InterpolationKernels for convenience.
    BSpline,
    BSplinePrime,
    CardinalCubicSpline,
    CardinalCubicSplinePrime,
    CatmullRomSpline,
    CatmullRomSplinePrime,
    CubicSpline,
    CubicSplinePrime,
    Kernel,
    LanczosKernel,
    LanczosKernelPrime,
    MitchellNetravaliSpline,
    MitchellNetravaliSplinePrime,
    # Other exports.
    AbstractInterpolator,
    AffineTransform,
    AffineTransform2D,
    BoundaryConditions,
    Flat,
    LazyMultidimInterpolator,
    SparseMultidimInterpolator,
    #SparseUnidimInterpolator,
    #LazyUnidimInterpolator,
    #interpolate,
    #interpolate!,
    #boundaries,
    promote_eltype,
    with_eltype

using InterpolationKernels

using TwoDimensional: AffineTransform2D

using LazyAlgebra
using LazyAlgebra.Foundations
import LazyAlgebra: apply, apply!, vcreate, output_size, input_size,
    coefficients

import Base: eltype, length, size, first, last, clamp, convert

include("AffineTransforms.jl")
import .AffineTransforms: AffineTransform, offset

include("types.jl")
include("utils.jl")
include("multidimensional.jl")
import .Multidimensional: LazyMultidimInterpolator, SparseMultidimInterpolator

include("fitting.jl")
import .Fitting: fit, solve, solve!

include("init.jl")

end # module
