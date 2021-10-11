#
# LinearInterpolators.jl -
#
# Provide interpolation methods as linear mappings.
#

module LinearInterpolators

# Export public types and methods.  Methods like `translate`, `compose`,
# `scale`, etc. which are accessible via operators like `+` or `*` or `∘` are
# however not exported.
export
    # Re-exports from TwoDimensional:
    AffineTransform2D,
    intercept,
    jacobian,
    rotate,

    # Re-exports from InterpolationKernels:
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
    iscardinal,
    isnormalized,

    Flat,
    Boundaries,
    LazyInterpolator,
    LazySeparableInterpolator,
    SafeFlat,
    SparseInterpolator,
    SparseSeparableInterpolator,
    TwoDimensionalTransformInterpolator,
    boundaries

using InterpolationKernels

using TwoDimensional.AffineTransforms
using TwoDimensional: AffineTransform2D

using LazyAlgebra
using LazyAlgebra.Foundations
import LazyAlgebra: apply, apply!, vcreate, output_size, input_size,
    coefficients

import Base: eltype, length, size, first, last, clamp, convert

# FIXME:
function rows end
# FIXME:
function columns end
# FIXME:
function fit end
# FIXME:
function regularize end
# FIXME:
function regularize! end

include("types.jl")
include("methods.jl")

#include("meta.jl")
#import .Meta
#include("tabulated.jl")
#using .TabulatedInterpolators
include("sparse.jl")
using .SparseInterpolators
include("lazy.jl")
#include("interpolate.jl")
#using .UnidimensionalInterpolators
#include("separable.jl")
#include("nonseparable.jl")

end # module
