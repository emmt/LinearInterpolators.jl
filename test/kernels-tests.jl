isdefined(:LazyInterpolators) || include("../src/LazyInterpolators.jl")

module LazyInterpolatorsKernelsTests


relabsdif(a::Ta,b::Tb) where {Ta<:Real,Tb<:Real} =
    relabsdif(float(promote_type(Ta,Tb)), a, b)

relabsdif(::Type{T}, a::Real, b::Real) where {T<:AbstractFloat} =
    relabsdif(T, T(a), T(b))

relabsdif(::Type{T}, a::T, b::T) where {T<:AbstractFloat} =
    (a == b ? zero(T) : abs(a - b)/max(abs(a),abs(b)))

maxrelerr(A, B) = maxrelerr(0.0, A, B)

function maxrelerr(err::T, A, B) where {T<:AbstractFloat}
    @assert length(A) == length(B)
    for (a,b) in zip(A,B)
        err = max(err, relabsdif(T, a, b))
    end
    return err
end

maxabserr(A, B) = maxabserr(0.0, A, B)

function maxabserr(err::T, A, B) where {T<:AbstractFloat}
    @assert length(A) == length(B)
    for (a,b) in zip(A,B)
        err = max(err, abs(T(a) - T(b)))
    end
    return err
end

print_result(success::Bool, value=success) =
    print_with_color((success ? :green : :red), value)

function print_maxabserror(ker, err; tol::Real=1e-15, pfx::String=" - ")
    print(pfx, summary(ker), ": max. abs. error = ")
    print_result(err < tol, @sprintf("%.3e\n", err))
end

function runtests()
    for (name, ker) in (("RectangularSpline", Kernels.RectangularSpline()),
                        ("RectangularSpline (type given)",
                         Kernels.RectangularSpline(Float32)),
                        #("Catmull-Rom",  Kernels.catmull_rom),
                        #("Mitchell-Netravali",  Kernels.mitchell_netravili),
                        ("cardinal Mitchell-Netravali",
                         Kernels.MitchellNetravaliSpline(Float32, 0, 1)),
                        ("Duff's tensioned B-spline",
                         Kernels.MitchellNetravaliSpline(Float32, 0.5, 0)),
                        ("Lanczos 4 interpolation function",
                         Kernels.LanczosKernel(4)))
        ker16 = ker(Float16)
        ker32 = ker(Float32)
        ker64 = ker(Float64)
        ker16a = Float16(ker)
        ker32a = Float32(ker)
        ker64a = Float64(ker)
        println(name," kernel:")
        println(" - support: ", length(ker))
        println(" - normalized: ", Kernels.isnormalized(ker))
        println(" - cardinal: ", Kernels.iscardinal(ker))
        println(" - ker(0): ", ker(0)) # test conversion
        print(" - ker32(0.1) ≈ ker64(0.1): ")
        print_result(ker32(0.1) ≈ ker64(0.1))
        println()
    end


    println("\nChecking weights:")
    offsets = (0.0, 0.1, 0.2, 0.3, 0.4)
    for ker in (lanczos2,)
        err = 0.0
        for t in offsets
            err = maxabserr(err, ker.(t .+ (0,-1)),
                            Kernels.getweights(ker, t))
        end
        print_maxabserror(ker, err)
    end
    for ker in (quadratic,)
        err = 0.0
        for t in offsets
            err = maxabserr(err, ker.(t .+ (1,0,-1)),
                            Kernels.getweights(ker, t))
        end
        print_maxabserror(ker, err)
    end

    for ker in (cubic, catmull_rom, mitchell_netravili, keys, lanczos4)
        err = 0.0
        for t in offsets
            err = maxabserr(err, ker.(t .+ (1,0,-1,-2)),
                            Kernels.getweights(ker, t))
        end
        print_maxabserror(ker, err)
    end
    for ker in (lanczos6,)
        err = 0.0
        for t in offsets
            err = maxabserr(err, ker.(t .+ (2,1,0,-1,-2,-3)),
                            Kernels.getweights(ker, t))
        end
        print_maxabserror(ker, err)
    end
    for ker in (lanczos8,)
        err = 0.0
        for t in offsets
            err = maxabserr(err, ker.(t .+ (3,2,1,0,-1,-2,-3,-4)),
                            Kernels.getweights(ker, t))
        end
        print_maxabserror(ker, err)
    end
end

runtests()

end # module
