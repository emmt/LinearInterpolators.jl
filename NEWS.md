# User visible changes for LinearInterpolators

## Branch v0.2

### Version 0.2.0

- Update to use the kernels provided by the new  branch (v0.2) of
  [`InterpolationKernels`](https://github.com/emmt/InterpolationKernels.jl).

- `SparseUnidimensionalInterpolator{T,S,D}` replaced by
  `SparseSeparableInterpolator{D,T,S}`.  Note the different order of type
  parameters which is intended to **avoid type instability**.

- `TabulatedInterpolator` replaced by `SparseSeparableInterpolator`.

- `apply(ker::Kernel, ...)` and `apply!(dst,ker::Kernel,...)` replaced by
  `interpolate` and `interpolate!` to **avoid type piracy** (the `apply` and
  `apply!` methods are imported from
  [`LazyAgebra`](https://github.com/emmt/LazyAlgebra.jl) while the kernel type
  of `ker` is provided by
  [`InterpolationKernels`](https://github.com/emmt/InterpolationKernels.jl)).

- New `LazyInterpolator` and `LazySeparableInterpolator` which compute
  interpolation coefficients **on the fly** and thus consume almost no memory
  even though they are not as fast as `SparseInterpolator` and
  `SparseSeparableInterpolator` if applied multiple times (e.g., in iterative
  methods).

- Interpolation coordinates are always specified as fractional indices in the
  interpolated array along the dimension of interpolation.  A range of indices
  may be specified to interpolate a contiguous sub-range of an array.


## Branch v0.1

### Version 0.1.3

- `SparseInterpolator(T,ker,...)` and
  `SparseUnidimensionalInterpolator(T,ker,...)` have been deprecated in favor
  of `SparseInterpolator{T}(ker,...)` and
  `SparseUnidimensionalInterpolator{T}(ker,...)`.

- Some documentation is available at https://emmt.github.io/LinearInterpolators.jl/dev


### Version 0.1.2

- `LinearInterpolators` is registered in personal registry
  [`EmmtRegistry`](https://github.com/emmt/EmmtRegistry).
