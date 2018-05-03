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

include("AffineTransforms.jl")
include("kernels.jl")
include("interpolations.jl")

end # module
