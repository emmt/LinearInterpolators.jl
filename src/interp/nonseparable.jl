#
# interp/nonseparable.jl --
#
# Non-separable multidimensional interpolation.
#
#------------------------------------------------------------------------------
#
# Copyright (C) 2017, Éric Thiébaut.
# This file is part of TiPi.  All rights reserved.
#

#------------------------------------------------------------------------------
# Direct operations.

function apply_direct!{T,S,B}(dst::Array{T,2},
                              ker::Kernel{T,S,B},
                              R::AffineTransform2D{T},
                              src::Array{T,2})
    apply_direct!(dst, ker, ker, R, src)
end

function apply_direct!{T<:AbstractFloat,B1<:Boundaries,B2<:Boundaries}(
    dst::Array{T,2}, ker1::Kernel{T,4,B1}, ker2::Kernel{T,4,B2},
    R::AffineTransform2D{T}, src::Array{T,2})

    # Get dimensions and limits.
    m1, m2 = size(dst)
    n1, n2 = size(src)
    lim1 = limits(ker1, n1)
    lim2 = limits(ker2, n2)

    # FIXME: if axis are aligned, use separable interpolation.

    # Apply the operator.
    for i2 in 1:m2
        pos2 = T(i2)
        off1 = R.xy*pos2 + R.x
        off2 = R.yy*pos2 + R.y
        @inbounds for i1 in 1:m1
            pos1 = T(i1)
            j11, j12, j13, j14,
            w11, w12, w13, w14 = getcoefs(ker1, lim1, R.xx*pos1 + off1)
            j21, j22, j23, j24,
            w21, w22, w23, w24 = getcoefs(ker2, lim2, R.yx*pos1 + off2)
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
# Adjoint operations.

function apply_adjoint!{T,S,B}(dst::Array{T,2},
                               ker::Kernel{T,S,B},
                               R::AffineTransform2D{T},
                               src::Array{T,2})
    apply_adjoint!(dst, ker, ker, R, src)
end

function apply_adjoint!{T<:AbstractFloat,B1<:Boundaries,B2<:Boundaries}(
    dst::Array{T,2}, ker1::Kernel{T,4,B1}, ker2::Kernel{T,4,B2},
    R::AffineTransform2D{T}, src::Array{T,2}; clr::Bool = true)

    # Get dimensions and limits.
    m1, m2 = size(src)
    n1, n2 = size(dst)
    lim1 = limits(ker1, n1)
    lim2 = limits(ker2, n2)

    # Apply adjoint operator.
    if clr
        fill!(dst, zero(T))
    end
    for i2 in 1:m2
        pos2 = T(i2)
        off1 = R.xy*pos2 + R.x
        off2 = R.yy*pos2 + R.y
        @inbounds for i1 in 1:m1
            pos1 = T(i1)
            j11, j12, j13, j14,
            w11, w12, w13, w14 = getcoefs(I1, R.xx*pos1 + off1)
            j21, j22, j23, j24,
            w21, w22, w23, w24 = getcoefs(I2, R.yx*pos1 + off2)
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
