#
# interp/unidimensional.jl --
#
# Unidimensional interpolation (the result may however be multi-dimensional).
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2018, Éric Thiébaut.
#

# All code is in a module to "hide" private methods.
module UnidimensionalInterpolators

using InterpolationKernels

using LazyAlgebra
import LazyAlgebra: apply, apply!, vcreate

using ...Interpolations
import ...Interpolations.Meta

#------------------------------------------------------------------------------
# Out-of place versions (coordinates cannot be a function).

function apply(ker::Kernel{T,S,<:Boundaries},
               x::AbstractArray{T,N},
               src::AbstractVector{T}) where {T,S,N}
    return apply(Direct, ker, x, src)
end

function apply(::Type{Direct}, ker::Kernel{T,S,<:Boundaries},
               x::AbstractArray{T,N},
               src::AbstractVector{T}) where {T,S,N}
    return apply!(Array{T}(undef, size(x)), Direct, ker, x, src)
end

function apply(::Type{Adjoint}, ker::Kernel{T,S,<:Boundaries},
               x::AbstractArray{T,N},
               src::AbstractArray{T,N}, len::Integer) where {T,S,N}
    return apply!(Array{T}(undef, len), Adjoint, ker, x, src)
end

#------------------------------------------------------------------------------
# In-place wrappers.

function apply!(dst::AbstractArray{T,N},
                ker::Kernel{T,S,<:Boundaries},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractVector{T}) where {T,S,N}
    apply!(1, Direct, ker, x, src, 0, dst)
end

function apply!(dst::AbstractArray{T,N},
                ::Type{Direct},
                ker::Kernel{T,S,<:Boundaries},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractVector{T}) where {T,S,N}
    apply!(1, Direct, ker, x, src, 0, dst)
end

function apply!(dst::AbstractVector{T},
                ::Type{Adjoint},
                ker::Kernel{T,S,<:Boundaries},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractArray{T,N}) where {T,S,N}
    apply!(1, Adjoint, ker, x, src, 0, dst)
end

#------------------------------------------------------------------------------
# In-place direct operation.

function __generate_interp(n::Integer, ker::Symbol, lim::Meta.Arg,
                           pos::Meta.Arg, arr::Symbol)
    J, W = Meta.make_varlist(:_j, n), Meta.make_varlist(:_w, n)
    code = Meta.generate_getcoefs(J, W, ker, lim, pos)
    expr = Meta.generate_interp_expr(arr, J, W)
    return (code, expr)
end

@generated function apply!(α::Real,
                           ::Type{Direct},
                           ker::Kernel{T,S,<:Boundaries},
                           x::AbstractArray{T,N},
                           src::AbstractVector{T},
                           β::Real,
                           dst::AbstractArray{T,N}) where {T,S,N}
    code, expr = __generate_interp(S, :ker, :lim, :pos, :src)
    quote
        @assert size(dst) == size(x)
        lim = limits(ker, length(src))
        if α == 0
            vscale!(dst, β)
        elseif β == 0
            if α == 1
                @inbounds for i in eachindex(dst, x)
                    pos = x[i]
                    $code
                    dst[i] = $expr
                end
            else
                alpha = convert(T, α)
                @inbounds for i in eachindex(dst, x)
                    pos = x[i]
                    $code
                    dst[i] = $expr*alpha
                end

            end
        else
            alpha = convert(T, α)
            beta = convert(T, β)
            @inbounds for i in eachindex(dst, x)
                pos = x[i]
                $code
                dst[i] = $expr*alpha + beta*dst[i]
            end
        end
        return dst
    end
end

@generated function apply!(α::Real,
                           ::Type{Direct},
                           ker::Kernel{T,S,<:Boundaries},
                           f::Function,
                           src::AbstractVector{T},
                           β::Real,
                           dst::AbstractArray{T,N}) where {T,S,N}
    code, expr = __generate_interp(S, :ker, :lim, :pos, :src)
    quote
        lim = limits(ker, length(src))
        if α == 0
            vscale!(dst, β)
        elseif β == 0
            if α == 1
                @inbounds for i in eachindex(dst)
                    pos = f(i) :: T
                    $code
                    dst[i] = $expr
                end
            else
                alpha = convert(T, α)
                @inbounds for i in eachindex(dst)
                    x = f(i) :: T
                    $code
                    dst[i] = $expr*alpha
                end
            end
        else
            alpha = convert(T, α)
            beta = convert(T, β)
            @inbounds for i in eachindex(dst)
                x = f(i) :: T
                $code
                dst[i] = $expr*alpha + beta*dst[i]
            end
        end
        return dst
    end
end

#------------------------------------------------------------------------------
# In-place adjoint operation.

function __generate_interp_adj(n::Integer, ker::Symbol, lim::Meta.Arg,
                               pos::Meta.Arg, dst::Symbol, val::Symbol)
    J, W = Meta.make_varlist(:_j, n), Meta.make_varlist(:_w, n)
    return (Meta.generate_getcoefs(J, W, ker, lim, pos),
            [:($dst[$(J[i])] += $(W[i])*$val) for i in 1:n]...)
end

@generated function apply!(α::Real,
                           ::Type{Adjoint},
                           ker::Kernel{T,S,<:Boundaries},
                           x::AbstractArray{T,N},
                           src::AbstractArray{T,N},
                           β::Real,
                           dst::AbstractVector{T}) where {T,S,N}
    code = __generate_interp_adj(S, :ker, :lim, :pos, :dst, :val)
    quote
        @assert size(src) == size(x)
        vscale!(dst, β)
        lim = limits(ker, length(dst))
        if α == 1
            @inbounds for i in eachindex(src, x)
                pos = x[i]
                val = src[i]
                $(code...)
        end
        elseif α != 0
            alpha = convert(T, α)
            @inbounds for i in eachindex(src, x)
                pos = x[i]
                val = alpha*src[i]
                $(code...)
            end
        end
        return dst
    end
end

@generated function apply!(α::Real,
                           ::Type{Adjoint},
                           ker::Kernel{T,S,<:Boundaries},
                           f::Function,
                           src::AbstractArray{T,N},
                           β::Real,
                           dst::AbstractVector{T}) where {T,S,N}
    code = __generate_interp_adj(S, :ker, :lim, :pos, :dst, :val)
    quote
        vscale!(dst, β)
        lim = limits(ker, length(dst))
        if α == 1
            @inbounds for i in eachindex(src)
                pos = f(i) :: T
                val = src[i]
                $(code...)
            end
        elseif α != 0
            alpha = convert(T, α)
            @inbounds for i in eachindex(src)
                pos = f(i) :: T
                val = alpha*src[i]
                $(code...)
            end
        end
        return dst
    end
end

end # module
