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
# Copyright (C) 2016-2022, Éric Thiébaut.
#

"""
    AbstractInterpolator{T,M,N}

is the abstract super-type of linear interpolators whose interpolation
coefficients have type `T` and which interpolates `N`-dimensional arrays to
produce `M`-dimensional arrays.

"""
abstract type AbstractInterpolator{T,M,N} <: LazyAlgebra.LinearMapping end

"""
    BoundaryConditions

abstract super-type of boundary conditions.  Boundary conditions define how
interpolated arrays are extrapolated.

See [`Flat`](@ref).

"""
abstract type BoundaryConditions end

"""
    Flat()

yields an instance of boundary conditions that assumes that extrapolated
positions correspond to the nearest position.

See [`AbstractBoundaryConditions`](@ref).

"""
struct Flat <: BoundaryConditions end

"""
    UndefinedType

is a type used to represent undefined type in `promote_type`.  It is an
abstract type so that `isbitstype(UndefinedType)` is false.

"""
abstract type UndefinedType end
