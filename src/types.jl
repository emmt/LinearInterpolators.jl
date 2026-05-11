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
    Undef

is the type of `undef`.

"""
const Undef = typeof(undef)

"""
    UndefinedType

Type used to represent undefined type in `promote_type`. It is an abstract type so that
`isbitstype(UndefinedType)` is false.

"""
abstract type UndefinedType end

"""
    LoopOrder{Outer,Inner}()

Return a singleton representing the ordering of the loops in operations involving 3 indices
(possibly multi-dimensional): `A[i1,i2,i3]` with, for example, `i1` and `i3` some spectator
indices and `i2` the running index. Parameter `Outer` specifies the rank of the index in the
outermost loop while `Inner` specifies the rank of the index in the innermost loop.

For example, with `LoopOrder{3,1}` (which is favorable for column-major storage order) the
code would write:

```julia
for i3 in I3 # <- outermost loop
    for i2 in I2
        for i1 in I1 # <- innermost loop
            # do something with A[i1,i2,i3]
        end
    end
end
```

But with `LoopOrder{3,2}`, the code would rather write:

```julia
for i3 in I3 # <- outermost loop
    for i1 in I1
        for i2 in I2 # <- innermost loop
            # do something with A[i1,i2,i3]
        end
    end
end
```

"""
struct LoopOrder{Outer,Inner} end

"""
    BoundaryConditions

Abstract super-type of boundary conditions. Boundary conditions define how interpolated
arrays are extrapolated.

See [`Flat`](@ref).

"""
abstract type BoundaryConditions end

"""
    Flat()

Return an instance of boundary conditions that assumes that extrapolated positions
correspond to the nearest position.

See [`BoundaryConditions`](@ref).

"""
struct Flat <: BoundaryConditions end

"""
    AbstractInterpolator{T,L,M,N}

Abstract super-type of linear interpolators whose interpolation coefficients have type `T`
and which interpolate `N` consecutive dimensions to produce `M` consecutive dimensions. Type
parameter `L` is the number of leading non-interpolated dimensions.

"""
abstract type AbstractInterpolator{T,L,M,N} <: LazyAlgebra.LinearMapping end

struct SparseInterpolator{T, # type of interpolation weights
                          L, # number of leading non-interpolated dimensions
                          M, # number of output interpolated dimensions
                          N, # number of input interpolated dimensions
                          O, # ordering of loops
                          S, # N-tuple of kernel lengths
                          W, # type of array of weights
                          I, # type of array of indices
                          } <: AbstractInterpolator{T,L,M,N}
    rows::Dims{M} # output interpolated dimensions
    cols::Dims{N} # input interpolated dimensions
    wgt::W # N-tuple of arrays of interpolation weights
    ind::I # N-tuple of arrays of interpolation weights
end

struct LazyInterpolator{T, # type of interpolation weights
                        L, # number of leading non-interpolated dimensions
                        M, # number of output interpolated dimensions
                        N, # number of input interpolated dimensions
                        O, # ordering of loops
                        S, # N-tuple of kernel lengths
                        P, # type of interpolation coordinates
                        K, # type of kernels
                        B, # type of boundary conditions
                        } <: AbstractInterpolator{T,L,M,N}
    rows::Dims{M} # output interpolated dimensions
    cols::Dims{N} # input interpolated dimensions
    pos::P # coordinates where to interpolate
    ker::K # N-tuple of interpolation kernels
    bnd::B # N-tuple of boundary conditions
end
