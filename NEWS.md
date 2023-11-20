# User visible changes for LinearInterpolators

## Version 0.1.8

- compatibility support for dependency TwoDimensional v0.4

## Version 0.1.7

- Extend `getcoefs` to cope with other coordinate types.  The coordinate type
  is converted on-the-fly if it is not exactly the floating-point type of the
  kernel.  The method `convert_coordinate` can be specialized to extend this
  conversion to non-standard numeric types.  This is automatically done for
  coordinates of type `Unitful.Quantity` when the `Unitful` package is loaded.


## Version 0.1.5

- Bug fixed.


## Version 0.1.3

- `SparseInterpolator(T,ker,...)` and
  `SparseUnidimensionalInterpolator(T,ker,...)` have been deprecated in favor
  of `SparseInterpolator{T}(ker,...)` and
  `SparseUnidimensionalInterpolator{T}(ker,...)`.

- Some documentation is available at https://emmt.github.io/LinearInterpolators.jl/dev


## Version 0.1.2

- `LinearInterpolators` is registered in personal registry
  [`EmmtRegistry`](https://github.com/emmt/EmmtRegistry).
