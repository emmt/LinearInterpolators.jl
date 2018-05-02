isdefined(:LazyInterpolators) || include("../src/LazyInterpolators.jl")

module LazyInterpolatorsAffineTransformsTests

using LazyInterpolators.AffineTransforms

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

distance(a::NTuple{2,Real}, b::NTuple{2,Real}) =
    hypot(a[1] - b[1], a[2] - b[2])

distance(A::AffineTransform2D, B::AffineTransform2D) =
    max(abs(A.xx - B.xx), abs(A.xy - B.xy), abs(A.x - B.x),
        abs(A.yx - B.yx), abs(A.yy - B.yy), abs(A.y - B.y))

@testset "AffineTransforms" begin
    tol = 1e-14
    I = AffineTransform2D()
    A = AffineTransform2D(1, 0, -3, 0.1, 1, +2)
    B = AffineTransform2D(-0.4, 0.1, -0.3, 0.7, 1.1, -0.9)
    C = inv(B)*B
    @test C.xx ≈ 1
    @test C.xy ≈ 0 atol=tol
    @test C.yx ≈ 0 atol=tol
    @test C.yy ≈ 1
    @test C.x  ≈ 0 atol=tol
    @test C.y  ≈ 0 atol=tol
    @test distance(C, I) ≤ tol
    @test det(I) == 1
    @test jacobian(A) == abs(det(A))
    @test distance(C, I) ≤ tol

    x0, y0 = 0.2, 1.3
    x1, y1 = B(x0, y0)
    x2, y2 = inv(B)(x1, y1)
    @test hypot(x2 - x0, y2 - y0) ≤ tol
    tx, ty = -2.3, 7.1
    K1 = B + (tx, ty); @test distance(K1(x0, y0), B(x0 + tx, y0 + ty)) ≤ tol
    K2 = (tx, ty) + B; @test distance(K2(x0, y0), B(x0, y0) .+ (tx, ty)) ≤ tol

    @testset "scaling" begin
        for α in (-1.7, 2.4, 0.1)
            @test distance((α*B)(x1, y1), α.*B(x1, y1)) ≤ tol
            @test distance((B*α)(x1, y1), B(α*x1, α*y1)) ≤ tol
        end
    end

    @testset "rotation" begin
        for θ in (-1.7, π/7, 0.1)
            R = rotate(+θ, I)
            Q = rotate(-θ, I)
            @test distance(R*Q, I) ≤ tol
            @test distance(Q*R, I) ≤ tol
            @test distance(rotate(θ, B)(x1, y1), R(B(x1, y1))) ≤ tol
            @test distance(rotate(B, θ)(x1, y1), B(R(x1, y1))) ≤ tol
        end
    end

    @testset "intercept" begin
        x, y = intercept(B)
        @test distance(B(x, y), (0,0)) ≤ tol
    end
end

end # module
