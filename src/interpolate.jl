#
# interpolate.jl --
#
# Unidimensional interpolation (the result may however be multi-dimensional).
#
#------------------------------------------------------------------------------
#
# This file is part of the LinearInterpolators package licensed under the MIT
# "Expat" License.
#
# Copyright (C) 2016-2021, Éric Thiébaut.
#

"""
    interpolate([P=Direct,] ker, x, src) -> dst

interpolates source array `src` at positions `x` with interpolation kernel
`ker` and yiedls the result as `dst`.

Interpolation is equivalent to applying a linear mapping.  Optional argument
`P` can be `Direct` or `Adjoint` to respectively compute the interpolation or
to apply the adjoint of the linear mapping implementing the interpolation.

"""
interpolate

"""
    interpolate!(dst, [P=Direct,] ker, x, src) -> dst

overwrites `dst` with the result of interpolating the source array `src` at
positions `x` with interpolation kernel `ker`.

Optional argument `P` can be `Direct` or `Adjoint`, see [`interpolate`](@ref) for
details.

"""
interpolate!

#------------------------------------------------------------------------------
# Out-of place versions (coordinates cannot be a function).

# FIXME: this is type-piracy

function interpolate(ker::Kernel{T,S},
               x::AbstractArray{T,N},
               src::AbstractVector{T}) where {T,S,N}
    return interpolate(Direct, ker, x, src)
end

function interpolate(::Type{Direct}, ker::Kernel{T,S},
               x::AbstractArray{T,N},
               src::AbstractVector{T}) where {T,S,N}
    return interpolate!(Array{T}(undef, size(x)), Direct, ker, x, src)
end

function interpolate(::Type{Adjoint}, ker::Kernel{T,S},
               x::AbstractArray{T,N},
               src::AbstractArray{T,N}, len::Integer) where {T,S,N}
    return interpolate!(Array{T}(undef, len), Adjoint, ker, x, src)
end

#------------------------------------------------------------------------------
# In-place wrappers.

function interpolate!(dst::AbstractArray{T,N},
                ker::Kernel{T,S},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractVector{T}) where {T,S,N}
    interpolate!(1, Direct, ker, x, src, 0, dst)
end

function interpolate!(dst::AbstractArray{T,N},
                ::Type{Direct},
                ker::Kernel{T,S},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractVector{T}) where {T,S,N}
    interpolate!(1, Direct, ker, x, src, 0, dst)
end

function interpolate!(dst::AbstractVector{T},
                ::Type{Adjoint},
                ker::Kernel{T,S},
                x::Union{Function,AbstractArray{T,N}},
                src::AbstractArray{T,N}) where {T,S,N}
    interpolate!(1, Adjoint, ker, x, src, 0, dst)
end

#------------------------------------------------------------------------------
# In-place direct operation.

function __generate_interp(n::Integer, ker::Symbol, lim::Meta.Arg,
                           pos::Meta.Arg, arr::Symbol)
    J = [Symbol(:j_,s) for s in 1:n]
    W = [Symbol(:w_,s) for s in 1:n]
    code = Meta.generate_getcoefs(J, W, ker, lim, pos)
    expr = Meta.generate_interp_expr(arr, J, W)
    return (code, expr)
end

@generated function interpolate!(α::Real,
                           ::Type{Direct},
                           ker::Kernel{T,S},
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

@generated function interpolate!(α::Real,
                           ::Type{Direct},
                           ker::Kernel{T,S},
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
    J = [Symbol(:j_,s) for s in 1:n]
    W = [Symbol(:w_,s) for s in 1:n]
    return (Meta.generate_getcoefs(J, W, ker, lim, pos),
            [:($dst[$(J[i])] += $(W[i])*$val) for i in 1:n]...)
end

@generated function interpolate!(α::Real,
                           ::Type{Adjoint},
                           ker::Kernel{T,S},
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

@generated function interpolate!(α::Real,
                                 ::Type{Adjoint},
                                 ker::Kernel{T,S},
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
