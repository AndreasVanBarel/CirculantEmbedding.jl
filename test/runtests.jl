println("Testing...")

using Test


##############################
# Testing CirculantEmbedding #
##############################
using CirculantEmbedding
using LinearAlgebra

## #####  NestedCirculant and NestedSymmetricCirculant #####

### NestedCirculant
A = [100i+10j+k for i in 1:4, j in 1:3, k in 1:2]
#A = [10j+k for j in 1:4, k in 1:3]
C = NestedCirculant(A)
@test C==[
111  411  311  211  131  431  331  231  121  421  321  221  112  412  312  212  132  432  332  232  122  422  322  222
211  111  411  311  231  131  431  331  221  121  421  321  212  112  412  312  232  132  432  332  222  122  422  322
311  211  111  411  331  231  131  431  321  221  121  421  312  212  112  412  332  232  132  432  322  222  122  422
411  311  211  111  431  331  231  131  421  321  221  121  412  312  212  112  432  332  232  132  422  322  222  122
121  421  321  221  111  411  311  211  131  431  331  231  122  422  322  222  112  412  312  212  132  432  332  232
221  121  421  321  211  111  411  311  231  131  431  331  222  122  422  322  212  112  412  312  232  132  432  332
321  221  121  421  311  211  111  411  331  231  131  431  322  222  122  422  312  212  112  412  332  232  132  432
421  321  221  121  411  311  211  111  431  331  231  131  422  322  222  122  412  312  212  112  432  332  232  132
131  431  331  231  121  421  321  221  111  411  311  211  132  432  332  232  122  422  322  222  112  412  312  212
231  131  431  331  221  121  421  321  211  111  411  311  232  132  432  332  222  122  422  322  212  112  412  312
331  231  131  431  321  221  121  421  311  211  111  411  332  232  132  432  322  222  122  422  312  212  112  412
431  331  231  131  421  321  221  121  411  311  211  111  432  332  232  132  422  322  222  122  412  312  212  112
112  412  312  212  132  432  332  232  122  422  322  222  111  411  311  211  131  431  331  231  121  421  321  221
212  112  412  312  232  132  432  332  222  122  422  322  211  111  411  311  231  131  431  331  221  121  421  321
312  212  112  412  332  232  132  432  322  222  122  422  311  211  111  411  331  231  131  431  321  221  121  421
412  312  212  112  432  332  232  132  422  322  222  122  411  311  211  111  431  331  231  131  421  321  221  121
122  422  322  222  112  412  312  212  132  432  332  232  121  421  321  221  111  411  311  211  131  431  331  231
222  122  422  322  212  112  412  312  232  132  432  332  221  121  421  321  211  111  411  311  231  131  431  331
322  222  122  422  312  212  112  412  332  232  132  432  321  221  121  421  311  211  111  411  331  231  131  431
422  322  222  122  412  312  212  112  432  332  232  132  421  321  221  121  411  311  211  111  431  331  231  131
132  432  332  232  122  422  322  222  112  412  312  212  131  431  331  231  121  421  321  221  111  411  311  211
232  132  432  332  222  122  422  322  212  112  412  312  231  131  431  331  221  121  421  321  211  111  411  311
332  232  132  432  322  222  122  422  312  212  112  412  331  231  131  431  321  221  121  421  311  211  111  411
432  332  232  132  422  322  222  122  412  312  212  112  431  331  231  131  421  321  221  121  411  311  211  111
]

# arithmetic: multiplication
v = rand(size(C,2))
prod0 = C*v
@test norm(prod0 - Matrix(C)*v) ≈ 0 atol=1e-10 #should be close to zero

# spectrum
# eigenvalues
islesscomplex(x::Complex,y::Complex) = abs(x)!=abs(y) ? abs(x)<abs(y) : imag(x)<imag(y)
λs = eigvals(C); λsorted = sort(λs; lt=islesscomplex, rev=true)
λs2 = eigvals(Matrix(C)); λ2sorted = sort(λs2; lt=islesscomplex, rev=true)
@test norm(λ2sorted - λsorted) ≈ 0 atol=1e-10 #should be close to zero

# eigenvectors
es = (eigvecs(C))
@test norm((Matrix(C)*es./es)[1,:]-eigvals(C)) ≈ 0 atol=1e-10 #should be close to zero
res = Matrix(C)*es - es*diagm(0=>eigvals(C))
@test norm(res) ≈ 0 atol=1e-10 #should be close to zero

### Padding and symmetrizing
symA = symmetrize(A)
NC = NestedCirculant(symA)
@test norm(NC-NC')==0 #should be exactly zero
NSC = NestedSymmetricCirculant(A)
@test norm(NSC-NC)==0 #should be exactly zero

### NestedSymmetricCirculant

# arithmetic: multiplication
v = ones(size(NSC,2))
prod1 = NSC*v
@test norm(prod1 - Matrix(NSC)*v) ≈ 0 atol=1e-10 #should be close to zero

# spectrum
# eigenvalues
λs = eigvals(NSC) # type should be Float64, not Complex
λsorted = sort(real(λs); rev=true)
λs2 = eigvals(Matrix(NSC)); λ2sorted = sort(λs2; rev=true)
@test norm(λ2sorted - λsorted) ≈ 0 atol=1e-10 #should be close to zero

# eigenvectors
es = (eigvecs(NSC))
@test norm((Matrix(NSC)*es./es)[1,:]-eigvals(NSC)) ≈ 0 atol=1e-10 #should be close to zero
res = Matrix(NSC)*es - es*diagm(0=>eigvals(NSC))
@test norm(res) ≈ 0 atol=1e-10 #should be close to zero

## ##### Stochastic Field Generation #####

λ = 0.3
σ = 0.5
p = 2
covfun = exponentialcovariance(λ,σ,p)
g = Gaussian(x->0,covfun)

grid = RegularGrid(9,9)
circulantembed(covfun,grid, print=1)

println("Testing the binary search for finding the optimal amount of padding in the circulant embedding method...")
grid = RegularGrid(257,257)
sampler = gen_sampler(g,grid, print=1)

# checking mean
println("Calculating and checking the mean of 1000 samples...")
function getμ(N::Int)
    sum = sampler(0)
    for i=1:N
        sum+=sampler(i)
    end
    return sum/(N+1)
end

@time μ = getμ(1000)
@test all(.≈(μ,0;atol=0.1))

# recovering the covariance function
println("Calculating and checking the covariance of 1000 samples...")
# Cov = 𝔼[(x-μ)(x-μ)ᵀ]
col = 23 #select the column of the covariance function to inspect
d = (sampler(0).-μ)
function getcov(N::Int)
    sum = d*d[col]'
    for i=1:N
        d .= sampler(i).-μ
        sum+=d*d[col]'
    end
    c = sum/(N+1)
end
@time c = getcov(1000)
c1 = reshape(c,size(grid))
exact_c1 = reshape([covfun(p,grid[col]) for p in grid],size(grid))
@test all(.≈(c1, exact_c1; atol=0.1))

### NOTE: fct is its own inverse, up to a scaling constant of prod(2.*size(X).-2)
R = rand(5,5)
c = prod(2.0.*size(R).-2)
@test norm(R-(CirculantEmbedding.fct(CirculantEmbedding.fct(R)))/c)≈0 atol=1e-13

