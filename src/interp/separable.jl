#
# interp/separable.jl --
#
# Separable Multidimensional interpolation.
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2021, Éric Thiébaut.
#

# FIXME: use a table for the innermost dimensions!

function apply!(dst::AbstractArray{T,2},
                ker::Kernel{T,S},
                x1::AbstractVector{T},
                x2::AbstractVector{T},
                src::AbstractArray{T,2}) where {T,S}
    apply!(1, Direct, ker, x1, ker, x2, src, 0, dst)
end

function apply!(dst::AbstractArray{T,2},
                ::Type{P},
                ker::Kernel{T,S},
                x1::AbstractVector{T},
                x2::AbstractVector{T},
                src::AbstractArray{T,2}) where {P<:Union{Direct,Adjoint},T,S}
    apply!(1, P, ker, x1, ker, x2, src, 0, dst)
end

function apply!(dst::AbstractArray{T,2},
                ker1::Kernel{T,S1},
                x1::AbstractVector{T},
                ker2::Kernel{T,S2},
                x2::AbstractVector{T},
                src::AbstractArray{T,2}) where {T,S1,S2}
    return apply!(1, Direct, ker1, x1, ker2, x2, src, 0, dst)
end

function apply!(dst::AbstractArray{T,2},
                ::Type{P},
                ker1::Kernel{T,S1},
                x1::AbstractVector{T},
                ker2::Kernel{T,S2},
                x2::AbstractVector{T},
                src::AbstractArray{T,2}) where {P<:Union{Direct,Adjoint},
                                                T,S1,S2}
    return apply!(1, P, ker1, x1, ker2, x2, src, 0, dst)
end

#------------------------------------------------------------------------------
# Direct operations.

@generated function apply!(α::Real,
                           ::Type{Direct},
                           ker1::Kernel{T,S1},
                           x1::AbstractVector{T},
                           ker2::Kernel{T,S2},
                           x2::AbstractVector{T},
                           src::AbstractArray{T,2},
                           β::Real,
                           dst::AbstractArray{T,2}) where {T,S1,S2}

    # Generate pieces of code.
    J1, W1 = Meta.make_varlist(:_j1, S1), Meta.make_varlist(:_w1, S1)
    J2, W2 = Meta.make_varlist(:_j2, S2), Meta.make_varlist(:_w2, S2)
    code1 = (Meta.generate_getcoefs(J1, W1, :ker1, :lim1, :(x1[i1])),)
    code2 = (Meta.generate_getcoefs(J2, W2, :ker2, :lim2, :(x2[i2])),
             [:( $(W2[i]) *= alpha ) for i in 1:S2]...)
    expr = Meta.generate_interp_expr(:src, J1, W1, J2, W2)

    quote
        # Get dimensions and limits.
        @assert size(dst) == (length(x1), length(x2))
        n1, n2 = size(dst)
        lim1 = limits(ker1, size(src, 1))
        lim2 = limits(ker2, size(src, 2))

        # Apply direct operator.
        if α == 0
            vscale!(dst, β)
        elseif β == 0
            alpha = convert(T, α)
            @inbounds for i2 in 1:n2
                $(code2...)
                for i1 in 1:n1
                    $(code1...)
                    dst[i1,i2] = $expr
                end
            end
        else
            alpha = convert(T, α)
            beta = convert(T, β)
            @inbounds for i2 in 1:n2
                $(code2...)
                for i1 in 1:n1
                    $(code1...)
                    dst[i1,i2] = $expr + beta*dst[i1,i2]
                end
            end
        end
        return dst
    end
end

#------------------------------------------------------------------------------
# In-place adjoint operation.

@generated function apply!(α::Real,
                           ::Type{Adjoint},
                           ker1::Kernel{T,S1},
                           x1::AbstractVector{T},
                           ker2::Kernel{T,S2},
                           x2::AbstractVector{T},
                           src::AbstractArray{T,2},
                           β::Real,
                           dst::AbstractArray{T,2}) where {T<:AbstractFloat,
                                                           S1,S2}

    # Generate pieces of code.
    J1, W1 = Meta.make_varlist(:_j1, S1), Meta.make_varlist(:_w1, S1)
    J2, W2 = Meta.make_varlist(:_j2, S2), Meta.make_varlist(:_w2, S2)
    code1 = [Meta.generate_getcoefs(J1, W1, :ker1, :lim1, :(x1[i1]))]
    code2 = [Meta.generate_getcoefs(J2, W2, :ker2, :lim2, :(x2[i2]))]
    temp = Meta.make_varlist(:_tmp, 1:S2)
    for i2 in 1:S2
        push!(code1, :( $(temp[i2]) = $(W2[i2])*val ))
        for i1 in 1:S1
            push!(code1, :(
                dst[$(J1[i1]),$(J2[i2])] += $(W1[i1])*$(temp[i2])
            ))
        end
    end

    quote
        # Get dimensions and limits.
        @assert size(src) == (length(x1), length(x2))
        n1, n2 = size(src)
        lim1 = limits(ker1, size(dst, 1))
        lim2 = limits(ker2, size(dst, 2))

        # Apply adjoint operator.
        vscale!(dst, β)
        if α != 0
            alpha = convert(T, α)
            @inbounds for i2 in 1:n2
                $(code2...)
                for i1 in 1:n1
                    val = src[i1,i2]*alpha
                    $(code1...)
                end
            end
        end
        return dst
    end
end
