# CirculantEmbedding.jl

Julia package for sampling Gaussian stochastic fields of arbitrary dimension using the circulant embedding method. The package also allows the so-called extended samples to be returned. By cutting or extracting different parts, these can be used to generate multiple weakly correlated samples of the stochastic field on the original small grid with just a single circulant embedding calculation. See Section 3.3 in [my PhD (2021)](https://lirias.kuleuven.be/retrieve/638063) for more information. 

## Installation

This is installed using the standard tools of the [package manager](https://julialang.github.io/Pkg.jl/v1/getting-started/):

```julia
pkg> add https://github.com/AndreasVanBarel/CirculantEmbedding.jl.git
```
You get the `pkg>` prompt by hitting `]` as the first character of the line.

## Usage

To load the module:

```julia
using CirculantEmbedding
```

The package is well documented. You can get started by 

```julia
help> CirculantEmbedding
```
where you get the `help>` prompt by hitting `?` as the first character of the line.

The following example outlines the basic usage of the package. We describe a Gaussian stochastic field and construct a sampler for it.

```julia
# Define the covariance function
λ = 0.3; σ = 0.5; p = 2;
covfun = exponentialcovariance(λ,σ,p)

# Define the Gaussian stochastic field
g = Gaussian(x->0, covfun)

# Construct a regular rectangular grid of points spanning [0,1]×[0,1]
grid = RegularGrid(257,257)

# Generate a sampler (with random seed)
sampler = gen_sampler(g, grid)

# Take a sample
sample = sampler(1)
```

It is also possible to generate samples on the full extended grid:

```julia
# Generate a sampler (this time with given seed)
sampler = gen_sampler(g, grid; seed=0, extendsamples=true)

# Take a sample
sample = sampler(1)

# Access to the extendedgrid
extendedgrid = sampler.extendedgrid
```

For testing purposes, one could plot the above sample, using, e.g., the Plots.jl package (not included as a dependency):

```julia
using Plots 
nodes_x = extendedgrid.ranges[1]
nodes_y = extendedgrid.ranges[2]
heatmap(nodes_x, nodes_y, sample')
surface(nodes_x, nodes_y, sample')
```

Stochastic fields of arbitrary dimension can be generated:

```julia
grid4d = RegularGrid(16,16,16,16) # 4d grid
sampler4d = gen_sampler(g, grid4d)
```

The package is efficient in the sense that it does not explicity generate or store the (nested) circulant matrices. However, for research purposes, this package allows one to inspect and play with these objects:

```julia
# Produce the circulant matrix that embeds the covariance matrix:
C, padding = circulantembed(covfun,grid; print=1) 
```

If `covfun` is an `IsotropicCovFun`, as is the case here, `C` will be a `NestedSymmetricCirculant`. If `covfun` is instead a mere `HomogeneousCovFun`, `C` will be a `NestedCirculant`.
