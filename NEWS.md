# User visible changes for LinearInterpolators

## Unreleased

- Module `LinearInterpolators.AffineTransforms` provides affine transforms to implement
  simple coordinate transforms for any number of dimensions. It supersedes the
  `AffineTranform2D` of the [`TwoDimensional`](https://github.com/emmt/TwoDimensional.jl)
  package which is limited to 2-D transforms.

- This version of `LinearInterpolators` is meant to work with version ≥ 0.2 of
  [`InterpolationKernels`](https://github.com/emmt/InterpolationKernels.jl). Boundaries
  conditions are now specified when building interpolators.

- Linear interpolators impose much less restrictions on the element types of their arguments
  (conversion of values is done on the fly). This allows for values with units such as those
  in the [`Unitful`](https://github.com/PainterQubits/Unitful.jl) package.

- Applying linear interpolators is faster because loop unrolling is performed along the
  kernel size (thanks to `@generated` code).

- The family of interpolators has been generalized and divided in two groups:

  - **Uni-dimensional interpolators** which interpolate along a single dimension of the
    input array, other dimensions of the output array matching those of the input array.
    Hence input and output arrays have aligned axes. Arbitrary geometric translations or
    resampling can be implemented by composing uni-dimensional operators, one for each
    dimension.

  - **Multi-dimensional interpolators** which interpolate all dimensions of the input array
    at a time. Input and output arrays may not have aligned axes and may have different
    number of dimensions. These operators can be used to implement interpolation for
    arbitrary coordinate transforms such as rotation, warping, etc.

- Interpolators of each group have two flavors, a **sparse** and a **lazy** one. **Sparse
  interpolators** have pre-computed interpolation coefficients. **Lazy interpolators**
  compute interpolation coefficients *on the fly*.

- The following table give some equivalences between the old and the new API:

  | Old                                                   | New                                           |
  |:------------------------------------------------------|:----------------------------------------------|
  | `LazyInterpolator(ker,pos)`                           | `LazyMultidimInterpolator(pos,ker,cols)`      |
  | `SparseInterpolator(ker,pos,rng)`                     | `SparseUnidimInterpolator(pos,ker,rng)`       |
  | `SparseUnidimensionalInterpolator(ker,d,pos,rng)`     | `SparseUnidimInterpolator(d,pos,ker,rng)`     |
  | `TwoDimensionalTransformInterpolator(rows,cols,ker,R` | `SparseMultidimInterpolator(R,rows,ker,cols)` |
  | `TabulatedInterpolator([d,]ker,pos,nrows,ncols)`      | `SparseUnidimensionalInterpolator`            |

  where `ker` is an interpolation kernel, `pos` is an array of interpolation positions,
  `rng` is an index range, `d` is a dimension rank, `rows` is the size of the output array,
  `cols` is the size of the input array, `R` is an instance of `AffineTranform2D`.


## Version 0.1.7

- Extend `getcoefs` to cope with other coordinate types.  The coordinate type
  is converted on-the-fly if it is not exactly the floating-point type of the
  kernel.  The method `convert_coordinate` can be specialized to extend this
  conversion to non-standard numeric types.  This is automatically done for
  coordinates of type `Unitful.Quantity` when the
  [`Unitful`](https://github.com/PainterQubits/Unitful.jl) package is loaded.


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
