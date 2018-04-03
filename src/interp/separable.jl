#
# interp/separable.jl --
#
# Separable Multidimensional interpolation.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2017-2018, Éric Thiébaut.
# This file is part of TiPi.  All rights reserved.
#

#------------------------------------------------------------------------------
# Direct operations.

function apply_direct!(dst::Array{T,2},
                       ker::Kernel{T,S,B},
                       x1::AbstractVector{T},
                       x2::AbstractVector{T},
                       src::AbstractArray{T,2}) where {T,S,B}
    apply_direct!(dst, ker, x1, ker, x2, src)
end

function apply_direct!(dst::Array{T,2},
                       ker1::Kernel{T,4,B1},
                       x1::AbstractVector{T},
                       ker2::Kernel{T,4,B2},
                       x2::AbstractVector{T},
                       src::AbstractArray{T,2}) where {T,B1,B2}
    # Get dimensions and limits.
    @assert size(dst) == (length(x1), length(x2))
    m1, m2 = size(dst)
    n1, n2 = size(src)
    lim1 = limits(ker1, n1)
    lim2 = limits(ker2, n2)

    # Apply direct operator.
    @inbounds for i2 in 1:m2
        j21, j22, j23, j24, w21, w22, w23, w24 = getcoefs(ker2, lim2, x2[i2])
        for i1 in 1:m1
            # FIXME: use a table for the innermost dimensions!
            j11, j12, j13, j14, w11, w12, w13, w14 = getcoefs(ker1, lim1, x1[i1])
            dst[i1,i2] = ((src[j11,j21]*w11 +
                           src[j12,j21]*w12 +
                           src[j13,j21]*w13 +
                           src[j14,j21]*w14)*w21 +
                          (src[j11,j22]*w11 +
                           src[j12,j22]*w12 +
                           src[j13,j22]*w13 +
                           src[j14,j22]*w14)*w22 +
                          (src[j11,j23]*w11 +
                           src[j12,j23]*w12 +
                           src[j13,j23]*w13 +
                           src[j14,j23]*w14)*w23 +
                          (src[j11,j24]*w11 +
                           src[j12,j24]*w12 +
                           src[j13,j24]*w13 +
                           src[j14,j24]*w14)*w24)
        end
    end
    return dst
end

#------------------------------------------------------------------------------
# In-place adjoint operation.

function apply_adjoint!(dst::Array{T,2},
                        ker::Kernel{T,S,B},
                        x1::AbstractVector{T},
                        x2::AbstractVector{T},
                        src::AbstractArray{T,2};
                        kwds...) where {T,S,B}
    apply_adjoint!(dst, ker, x1, ker, x2, src; kwds...)
end

function apply_adjoint!(dst::Array{T,2},
                        ker1::Kernel{T,4,B1},
                        x1::AbstractVector{T},
                        ker2::Kernel{T,4,B2},
                        x2::AbstractVector{T},
                        src::AbstractArray{T,2};
                        clr=true) where {T,B1,B2}
    # Get dimensions and limits.
    @assert size(src) == (length(x1), length(x2))
    m1, m2 = size(src)
    n1, n2 = size(dst)
    lim1 = limits(ker1, n1)
    lim2 = limits(ker2, n2)

    # Apply adjoint operator.
    if clr
        fill!(dst, zero(T))
    end
    @inbounds for i2 in 1:m2
        j21, j22, j23, j24, w21, w22, w23, w24 = getcoefs(ker2, lim2, x2[i2])
        for i1 in 1:m1
            # FIXME: use a table for the innermost dimensions!
            j11, j12, j13, j14, w11, w12, w13, w14 = getcoefs(ker1, lim1, x1[i1])
            a = src[i1,i2]
            w21a = w21*a
            dst[j11,j21] += w11*w21a
            dst[j12,j21] += w12*w21a
            dst[j13,j21] += w13*w21a
            dst[j14,j21] += w14*w21a
            w22a = w22*a
            dst[j11,j22] += w11*w22a
            dst[j12,j22] += w12*w22a
            dst[j13,j22] += w13*w22a
            dst[j14,j22] += w14*w22a
            w23a = w23*a
            dst[j11,j23] += w11*w23a
            dst[j12,j23] += w12*w23a
            dst[j13,j23] += w13*w23a
            dst[j14,j23] += w14*w23a
            w24a = w24*a
            dst[j11,j24] += w11*w24a
            dst[j12,j24] += w12*w24a
            dst[j13,j24] += w13*w24a
            dst[j14,j24] += w14*w24a
        end
    end
    return dst
end
