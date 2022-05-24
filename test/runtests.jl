module TestingLinearInterpolators

using Test

using LazyAlgebra
using LazyAlgebra: Adjoint, Direct

using LinearInterpolators
using LinearInterpolators: compute_indices, nth, offset, UndefinedType

using InterpolationKernels
using StaticArrays

@testset "Utilities        " begin
    # Undefined type.
    @test promote_type(UndefinedType) === UndefinedType
    @test promote_type(UndefinedType, UndefinedType) === UndefinedType
    @test promote_type(UndefinedType, Any) === Any
    @test promote_type(UndefinedType, Int8) === Int8
    @test promote_type(Any, UndefinedType) === Any
    @test promote_type(Int8, UndefinedType) === Int8

    # Promote element type.
    @test promote_eltype() === LinearInterpolators.UndefinedType
    @test promote_eltype(Array{Float32}) === Float32
    @test promote_eltype([1]) === Int
    @test promote_eltype(1.0) === Float64
    @test promote_eltype(CatmullRomSpline{Float64}) === Float64
    @test promote_eltype(Array{Float32}, [0], 1.0) === Float64
    @test promote_eltype(Array{Int16}, CatmullRomSpline{Float32}()) === Float32

    # Convert element type.
    A = Float32[1 2 3; 4 5 5]
    B = with_eltype(Int16(3), A)
    @test eltype(B) === Int16
    @test size(B) === size(A)
    @test B == A
    @test with_eltype(Float32, CatmullRomSpline{Float64}()) ===
        CatmullRomSpline{Float32}()

    # Ordinal numbers.
    @test nth(0) == "0-th"
    @test nth(10) == "10-th"
    @test nth(1) == "1-st"
    @test nth(21) == "21-st"
    @test nth(2) == "2-nd"
    @test nth(32) == "32-nd"
    @test nth(3) == "3-rd"
    @test nth(43) == "43-rd"
    @test nth(4) == "4-th"
    @test nth(54) == "54-th"

    # Computation of interpolation indices.
    for len in (1, 2, 7)
        for off in -5.0:1.0:len+1.0
            @test compute_indices(Flat(), off, len, Val(4)) ===
                ntuple(i -> clamp(i + Int(off), 1, len), Val(4))
        end
    end
end

@testset "Affine Transforms" begin
    # 3-dimensional -> 2-dimensional (non-invertible) transform.
    A1 = SMatrix{2,3}(1,2,3,4,5,6)
    b1 = SVector(7,8)
    R1 = AffineTransform(A1, b1)
    @test R1 isa AffineTransform{2,3}
    @test eltype(R1) === promote_type(eltype(A1), eltype(b1))
    @test with_eltype(Float32, R1) == R1
    @test eltype(with_eltype(Int16, R1)) === Int16
    @test axes(R1) == (1:2, 0:3)
    @test size(R1) == (2, 4)
    @test length(R1) == 8
    @test Tuple(R1) === (7,1,3,5, 8,2,4,6)
    @test offset(R1) === b1.data
    @test offset(R1) === R1*(0,0,0)
    @test offset(R1) === R1(0,0,0)
    @test offset(R1) === (R1*SVector(0,0,0)).data
    @test SVector(R1) === b1
    @test Vector(R1) == b1
    @test SMatrix(R1) === A1
    @test Matrix(R1) == A1
    # 3-dimensional -> 3-dimensional (invertible) transform with integer
    # coefficients such that inverse also have integer coefficients.
    A2 = SMatrix{3,3}(1,2,2,1,2,1,1,1,2)
    b2 = SVector(3,4,5)
    R2 = AffineTransform(A2, b2)
    @test Tuple(inv(R2)) == (0, -3, 1, 1, -1, 2, 0, -1, -2, 2, -1, 0)
    @test R2\b2 == SVector(0,0,0)
    x = (-1,2,-3)
    y = R2*x
    @test y == (1, 3, -1)
    @test R2\y == x
    # Compose transforms.
    R3 = R1*R2
    @test SMatrix(R3) === A1*A2
    @test SVector(R3) === A1*b2 + b1
end

end # module
