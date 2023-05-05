using RegularGrids #for definition of the regular rectangular grids
using CirculantEmbedding

# Covariance
λ = 0.3
σ = 0.5
p = 2
covfun = exponentialcovariance(λ,σ,p)

# Stochastic field
g = Gaussian(x->0, covfun)

# Grid
grid = RegularGrid(257,257)

# Embedding in a circulant matrix
C, padding = circulantembed(covfun,grid; print=1) 

# Producing sampler
sampler = gen_sampler(g, grid; extendsamples=true)

# Taking a sample
sample = sampler(1)
extendedgrid = sampler.extendedgrid

## Plotting
using Plots 
nodes_x = extendedgrid.ranges[1]
nodes_y = extendedgrid.ranges[2]
heatmap(nodes_x, nodes_y, sample')
surface(nodes_x, nodes_y, sample')
