isdefined(:LazyInterpolators) || include("../src/LazyInterpolators.jl")

module LazyInterpolatorsKernelsTests

using LazyInterpolators.Kernels

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

@testset "Kernels" begin
    offsets = (0.0, 0.1, 0.2, 0.3, 0.4)
    tol = 1e-14
    for (nam, ker, sup, nrml, card) in (
        ("Box",                         RectangularSpline(),              1, true,  true),
        ("Triangle",                    LinearSpline(),                   2, true,  true),
        ("Quadratic B-spline",          QuadraticSpline(),                3, true,  false),
        ("Cubic B-spline",              CubicSpline(),                    4, true,  false),
        ("Catmull-Rom spline",          CatmullRomSpline(),               4, true,  true),
        ("Cardinal cubic spline",       CardinalCubicSpline(-1),          4, true,  true),
        ("Mitchell & Netravali spline", MitchellNetravaliSpline(),        4, true,  false),
        ("Duff's tensioned B-spline",   MitchellNetravaliSpline(0.5, 0),  4, true,  false),
        ("Keys's (emulated)",           MitchellNetravaliSpline(0, -0.7), 4, true,  true),
        ("Keys's cardinal cubics",      KeysSpline(-0.7),                 4, true,  true),
        ("Lanczos 4 kernel",            LanczosKernel(4),                 4, false, true),
        ("Lanczos 6 kernel",            LanczosKernel(6),                 6, false, true))
        @testset "$nam" begin
            @test isnormalized(ker) == nrml
            @test iscardinal(ker) == card
            @test length(ker) == sup
            @test eltype(ker) == Float64
            @test eltype(Float32(ker)) == Float32
            @test boundaries(ker) == Flat
            if iscardinal(ker)
                @test ker(0) == 1
                @test maximum(abs.(ker([-3,-2,-1,1,2,3]))) ≤ tol
            end

            # S is the tuple of shifts applied in getweights.
            s = ntuple(i -> ((sup + 1) >> 1) - i, sup)
            err = 0.0
            for t in offsets
                dif = ker.(t .+ s) .- Kernels.getweights(ker, t)
                err = max(err, maximum(abs.(dif)))
            end
            @test err ≤ tol
        end
    end

end

end # module
