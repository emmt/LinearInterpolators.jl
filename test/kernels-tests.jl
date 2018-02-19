module TiPiTests

const PLOTTING = true
if PLOTTING
    using PyCall; pygui(:gtk); # can be: :wx, :gtk, :qt
    using LaTeXStrings
    import PyPlot; const plt = PyPlot
end
include("../src/kernels.jl")

function runtests()
    for (name, ker) in (("RectangularSpline", Kernels.RectangularSpline()),
                        ("RectangularSpline (type given)",
                         Kernels.RectangularSpline(Float32)),
                        #("Catmull-Rom",  Kernels.catmull_rom),
                        #("Mitchell-Netravali",  Kernels.mitchell_netravili),
                        ("cardinal Mitchell-Netravali",
                         Kernels.MitchellNetraviliSpline(Float32, 0, 1)),
                        ("Duff's tensioned B-spline",
                         Kernels.MitchellNetraviliSpline(Float32, 0.5, 0)),
                        ("Lanczos 4 interpolation function",
                         Kernels.LanczosKernel(4)))
        ker16 = ker(Float16)
        ker32 = ker(Float32)
        ker64 = ker(Float64)
        println(name," kernel:")
        println(" - support: ", length(ker))
        println(" - normalized: ", Kernels.isnormalized(ker))
        println(" - cardinal: ", Kernels.iscardinal(ker))
        println(" - ker(0): ", ker(0)) # test conversion
        println(" - ker32(0.1) ≈ ker64(0.1): ", ker32(0.1) ≈ ker64(0.1))
    end

    box = Kernels.RectangularSpline()
    triangle = Kernels.LinearSpline()
    quadratic = Kernels.QuadraticSpline()
    cubic = Kernels.CubicSpline()
    catmull_rom = Kernels.CatmullRomSpline()
    mitchell_netravili = Kernels.MitchellNetraviliSpline()
    keys = Kernels.KeysSpline(0.5)
    lanczos2 = Kernels.LanczosKernel(2)
    lanczos4 = Kernels.LanczosKernel(4)
    lanczos6 = Kernels.LanczosKernel(6)

    plt.figure(1)
    plt.clf()
    x = linspace(-5,5,1000);
    plt.plot(x, box(x), color="darkgreen",
             linewidth=2.0, linestyle="-");
    plt.plot(x, triangle(x), color="darkblue",
             linewidth=2.0, linestyle="-");
    plt.plot(x, quadratic(x), color="darkcyan",
             linewidth=2.0, linestyle="-");
    plt.plot(x, cubic(x), color="firebrick",
             linewidth=2.0, linestyle="-");
    plt.plot(x, catmull_rom(x), color="orange",
             linewidth=2.0, linestyle="-");
    plt.plot(x, mitchell_netravili(x), color="violet",
             linewidth=2.0, linestyle="-");
    plt.plot(x, lanczos6(x), color="black",
             linewidth=2.0, linestyle="-");
    plt.title("Some kernel functions");

    for ker in (lanczos2,)
        for t in (0.0, 0.1, 0.2, 0.2, 0.4)
            println(typeof(ker), ": ",
                    ker.(t .+ (0,-1)) .≈ Kernels.getweights(ker, t))
        end
    end
    for ker in (quadratic,)
        for t in (0.0, 0.1, 0.2, 0.2, 0.4)
            println(typeof(ker), ": ",
                    ker.(t .+ (1,0,-1)) .≈ Kernels.getweights(ker, t))
        end
    end

    for ker in (cubic, catmull_rom, mitchell_netravili, keys, lanczos4)
        for t in (1e-6, 0.1, 0.2, 0.2, 0.4)
            println(typeof(ker), ": ",
                    ker.(t .+ (1,0,-1,-2)) .≈ Kernels.getweights(ker, t))
        end
    end
    for ker in (lanczos6,)
        for t in (1e-6, 0.1, 0.2, 0.2, 0.4)
            println(typeof(ker), ": ",
                    ker.(t .+ (2,1,0,-1,-2,-3)) .≈ Kernels.getweights(ker, t))
        end
    end
end

runtests()

end # module
