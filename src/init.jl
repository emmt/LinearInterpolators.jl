using Requires

function __init__()
    @require Unitful="1986cc42-f94f-5a68-af5c-568840ba703d" begin
        # FIXME: Should be restricted to dimensionless quantities because the
        # argument of a kernel function is in units of the interpolation grid
        # step size.
        convert_coordinate(T::Type{<:AbstractFloat}, c::Unitful.Quantity) =
            convert(T, c.val)
    end
end
