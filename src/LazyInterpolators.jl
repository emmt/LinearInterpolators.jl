#
# LazyInterpolators.jl -
#
# Implement various interpolation methods as linear mappings.
#
#------------------------------------------------------------------------------
#
# This file is part of the LazyInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2018, Éric Thiébaut.
#

module LazyInterpolators

include("AffineTransforms.jl")
include("kernels.jl")
include("interpolations.jl")

end # module
