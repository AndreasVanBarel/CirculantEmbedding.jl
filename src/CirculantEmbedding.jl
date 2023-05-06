"""
    CirculantEmbeddding

Module for sampling stochastic fields using the circulant embedding (CE) method. 

Most important function is [`CirculantEmbed.gen_sampler`](@ref), which yields a function that can sample a stochastic field with given properties on a given regular rectangular grid.
"""
module CirculantEmbedding

# Points
include("Points.jl")
using .Points 
export Points
export Point

# RegularGrids
include("RegularGrids.jl")
using .RegularGrids
export RegularGrids
export RegularGrid
export getΔ
export extend

# CirculantEmbedding
using FFTW
using Random
using LinearAlgebra
using SpecialFunctions # gamma function in Matern covariance

import Base: eltype, ndims, size, length, axes, getindex, show, +, -, *, /, \
import LinearAlgebra: eigen, eigvals, eigvecs

export StochField, Gaussian, LogNormal, Uniform, Deterministic
export CovFun, HomogeneousCovFun, IsotropicCovFun, GeneralCovFun
export exponentialcovariance, materncovariance, rationalquadraticcovariance
export gen_sampler
export ANestedCirculant, NestedCirculant, NestedSymmetricCirculant
export covariancematrix, circulantembed, getblockstructure
export eigvec
export symmetrize

function __init__()
    FFTW.set_num_threads(8)
end

###########################
### COVARIANCE FUNCTION ###
###########################

"""
    CovFun

Abstract type representing all covariance functions. 
The covariance represented by a `c::CovFun` between a pair of points `p1::Point{N}` and `p2::Point{N}` for some `N` can be evaluated by `c(p1,p2)`.

Some specialized covariance functions can additionally be evaluated using specialized methods, see [`HomogeneousCovFun`](@ref), [`IsotropicCovFun`](@ref).
For a CovFun that does not assume any special structure, see [`GeneralCovFun`](@ref).
"""
abstract type CovFun end

"""
    GeneralCovFun

A covariance function that does not assume any special structure. It should be created by `GeneralCovFun(f)` where `f` is any function 

    f(p1::Point{N}, p2::Point{N})->Float64.

Evaluating some `c::GeneralCovFun`is done using `c(p1::Point{N}, p2::Point{N})`.

If `f` has structure that the subsequent algorithms should exploit, use [`HomogeneousCovFun`](@ref) or [`IsotropicCovFun`](@ref) instead.
"""
struct GeneralCovFun <: CovFun
    fun::Function
end

"""
    HomogeneousCovFun

Represents a covariance function between points p1 and p2 that is dependent only on Δp = p1-p2. 
It should be created by `HomogeneousCovFun(f)` where `f` is any function 

    f(Δp::Point{N})->Float64

Alternatively, a given `covfun::IsotropicCovFun` can be converted to a `HomogeneousCovFun` by 

    HomogeneousCovFun(covfun)

This is handy if one wishes to exploit only the homogeneity but not the isotropy of `covfun` in subsequent calculations.

Evaluating some `c::HomogeneousCovFun` is done using `c(p1::Point{N},p2::Point{N})` or `c(Δp::Point{N})`.

See also [`IsotropicCovFun`](@ref).
"""
struct HomogeneousCovFun <: CovFun #Homogeneous : only dependent on x-y
    fun::Function
end

"""
    IsotropicCovFun

Represents a covariance function between points p1 and p2 that is dependent only on δ = ‖p1-p2‖ with ‖⋅‖ some norm that is invariant under sign changes in any one element of its input. Note that this is somewhat less restrictive than what is usually understood as an isotropic cost function. Also note that all isotropic covariance functions are also mathematically homogeneous. If only the exploitation of the homogeneous structure is desired, one should create a `HomogeneousCovFun` instead.

An `IsotropicCovFun` is created by `IsotropicCovFun(f, n)` where `f` is any function 

    f(δ::Float64)->Float64,

and `n(Δp::Point{N})->Float64` is some norm.

Evaluating some `c::IsotropicCovFun` is done using `c(p1::Point{N},p2::Point{N})` or `c(Δp::Point{N})` or `c(δ::Float64)`.

See also [`HomogeneousCovFun`](@ref).
"""
struct IsotropicCovFun <: CovFun #Isotropic : only dependent on ‖x-y‖ with ‖⋅‖ some norm that is invariant under sign changes in any one element of its input
    fun::Function
    norm::Function
end

HomogeneousCovFun(covfun::IsotropicCovFun) = HomogeneousCovFun(covfun.fun ∘ covfun.norm)

# show(io::IO, h::GeneralCovFun) = print(io, "a GeneralCovFun")
# show(io::IO, h::HomogeneousCovFun) = print(io, "a HomogeneousCovFun")
# show(io::IO, h::IsotropicCovFun) = print(io, "an IsotropicCovFun")
show(io::IO, h::GeneralCovFun) = print(io, "GeneralCovFun $(h.fun)")
show(io::IO, h::HomogeneousCovFun) = print(io, "HomogeneousCovFun $(h.fun)")
show(io::IO, h::IsotropicCovFun) = print(io, "IsotropicCovFun $(h.fun) with norm $(h.norm)")

(K::GeneralCovFun)(ps...) = K.fun(ps...)::Real
(K::HomogeneousCovFun)(Δ) = K.fun(Δ)::Real
(K::HomogeneousCovFun)(x,y) = K(x-y)::Real
(K::IsotropicCovFun)(δ::Real) = K.fun(δ)::Real
(K::IsotropicCovFun)(Δ) = K(K.norm(Δ))::Real
(K::IsotropicCovFun)(x,y) = K(x-y)::Real

"""
    exponentialcovariance(λ::Real, σ::Real, norm)->IsotropicCovFun

  * `λ` - correlation length.
  * `σ` - standard deviation.
  * `norm` - the norm represented in the formula below by ‖⋅‖.
Returns an `IsotropicCovFun` representing δ -> σ²exp(-δ/λ) and Δp -> σ²exp(-‖Δp‖/λ)
"""
exponentialcovariance(λ::Real, σ::Real, norm) = IsotropicCovFun(δ->σ^2*exp(-δ/λ), norm)

"""
    exponentialcovariance(λ::Real, σ::Real, p::Real=2)->IsotropicCovFun

Equivalent to exponentialcovariance(λ, σ, Δp->norm(Δp,p)), i.e., p demands the p-norm be used.
"""
exponentialcovariance(λ::Real, σ::Real, p::Real=2) = IsotropicCovFun(δ->σ^2*exp(-δ/λ), Δp->norm(Δp,p))

"""
    rationalquadraticcovariance(α::Real, λ::Real, σ::Real, norm)->IsotropicCovFun

  * `α` - exponent.
  * `λ` - correlation length.
  * `σ` - standard deviation
  * `norm` - the norm represented in the formula below by ‖⋅‖.
Returns an `IsotropicCovFun` representing δ -> σ²(1+δ²/2αλ²)^(-α) and Δp -> σ²(1+‖Δp‖²/2αλ²)^(-α)
"""
rationalquadraticcovariance(α::Real, λ::Real, σ::Real, norm) = IsotropicCovFun(δ->σ^2*(1+(δ/λ)^2/(2α))^-α, norm)

"""
    rationalquadraticcovariance(α::Real, λ::Real, σ::Real, p::Real=2)->IsotropicCovFun

Equivalent to rationalquadraticcovariance(α, λ, σ, Δp->norm(Δp,p)), i.e., p demands the p-norm be used.
"""
rationalquadraticcovariance(α::Real, λ::Real, σ::Real, p::Real=2) = IsotropicCovFun(δ->σ^2*(1+(δ/λ)^2/(2α))^-α, Δp->norm(Δp,p))

"""
    materncovariance(v::Real, λ::Real, σ::Real, norm)->IsotropicCovFun

  * `ν` - smoothness parameter.
  * `λ` - correlation length.
  * `σ` - standard deviation
  * `norm` - the norm represented in the formula below by ‖⋅‖.
Returns an `IsotropicCovFun` representing the Matérn coveriance function, see [Wikipedia](https://en.wikipedia.org/wiki/Matérn_covariance_function), with parameters as described above.
"""
function materncovariance(v::Real, λ::Real, σ::Real, norm) #v is a smoothness parameter, λ is a correlation length metric
    isinf(v) && ( return IsotropicCovFun(Δ->σ^2*exp(-(Δ/λ)^2/2)))
    Γ = gamma; Kᵥ(x) = besselk(v,x);
    function materncovfun(Δp)
        δ = norm(Δp)
        if δ>1e-8 # normal evaluation
            return σ^2*2.0^(1-v)/Γ(v)*(√(2v)δ/λ)^v*Kᵥ(√(2v)δ/λ)
        else
            return σ^2 # first term of the Taylor series expansion
            #return σ^2*(1 + v/(2-2v)*(d/ρ)^2 + v^2/(8(2-3v+v^2))*(d/ρ)^4)  # fourth order Taylor expansion
        end
    end
    IsotropicCovFun(materncovfun)
end

"""
    materncovariance(v::Real, λ::Real, σ::Real, p::Real=2)->IsotropicCovFun

Equivalent to materncovariance(ν, λ, σ, Δp->norm(Δp,p)), i.e., p demands the p-norm be used.
"""
materncovariance(v::Real, λ::Real, σ::Real, p::Real=2) = materncovariance(v, λ, σ, Δp->norm(Δp,p))

#########################
### COVARIANCE MATRIX ###
#########################
"""
    covariancematrix(covfun::CovFun, grid::RegularGrid)->Matrix

Naively calculates and constructs the covariance matrix for the given `grid`, storing it as a dense `Matrix`. Note that, depending on `covfun`, the resulting matrix is not guaranteed to be positive definite!
"""
covariancematrix(covfun::CovFun, grid::RegularGrid) = reshape([covfun(p1,p2) for p1 in grid, p2 in grid],length(grid),length(grid))
"""
    covariancematrix(covfun::CovFun, points::Array)->Matrix

Naively calculates and constructs the covariance matrix for the given `Array` of points, storing it as a dense `Matrix`. Note that, depending on `covfun`, the resulting matrix is not guaranteed to be positive definite!
"""
covariancematrix(covfun::CovFun, points::Array) = reshape([covfun(p1,p2) for p1 in points, p2 in points],length(points),length(points))

########################
### STOCHASTIC FIELD ###
########################
"""
    StochField

Represents a stochastic field, i.e., a random function. 

See also [`Gaussian`](@ref), [`LogNormal`](@ref), [`Uniform`](@ref), and [`Deterministic`](@ref).
"""
abstract type StochField end

"""
    Gaussian <: StochField

`Gaussian(mean::Function, covfun::CovFun)` represents a Gaussian stochastic field with mean function `mean` and covariance function `covfun`.

`Gaussian(covfun::Covfun)` is a shorthand for `Gaussian(x->0, covfun)`.

See also [`LogNormal`](@ref), [`Uniform`](@ref), and [`Deterministic`](@ref).
"""
struct Gaussian <: StochField
    mean::Function
    covfun::CovFun
end
Gaussian(c::CovFun) = Gaussian(x->0,c)

"""
    LogNormal <: StochField

`LogNormal(mean::Function, covfun::CovFun)` represents a LogNormal stochastic field with underlying Gaussian distribution with mean function `mean` and covariance function `covfun`.

`LogNormal(covfun::Covfun)` is a shorthand for `LogNormal(x->0, covfun)`.

See also [`Gaussian`](@ref), [`Uniform`](@ref), and [`Deterministic`](@ref).
"""
struct LogNormal <: StochField
    mean::Function
    covfun::CovFun
end
LogNormal(c::CovFun) = LogNormal(x->0,c)

"""
    Uniform <: StochField

`Uniform(lowest, highest)` represents a Uniform stochastic field. Samples are constant functions with a constant uniformly distributed between `lowest` and `highest`.

`Uniform()` is a shorthand for `Uniform(0.0, 1.0)`.

See also [`Gaussian`](@ref), [`LogNormal`](@ref), and [`Deterministic`](@ref).
"""
struct Uniform <: StochField
    lowest::Float64
    highest::Float64
end
Uniform() = Uniform(0.0, 1.0)

"""
    Deterministic <: StochField

`Deterministic(fun::Function)` represents the deterministic function `fun`, i.e., a stochastic field where all probability density is allocated to the single function `fun`.

`Deterministic(value::Number)` is a shorthand for `Deterministic(x->value)`.

See also [`Gaussian`](@ref), [`LogNormal`](@ref), and [`Uniform`](@ref).
"""
struct Deterministic <: StochField
    fun::Function
end
Deterministic(value::Number) = Deterministic(x->value)

show(io::IO, sf::Gaussian) = print(io, "Gaussian field with mean $(sf.mean) and covariance $(sf.covfun)")
show(io::IO, sf::LogNormal) = print(io, "LogNormal field e^z with z Gaussian with mean $(sf.mean) and covariance $(sf.covfun)")
show(io::IO, sf::Uniform) = print(io, "Uniform field on [$(sf.lowest),$(sf.highest)]")
show(io::IO, sf::Deterministic) = print(io, "Deterministic field equal to $(sf.fun)")

###########################
### CIRCULANT EMBEDDING ###
###########################
"""
    circulantembed(covfun, grid; tol=1e-13, maxi=10000, approx=false, print=false)

  * `covfun` - A `HomogeneousCovFun` or `IsotropicCovFun`.
  * `grid::RegularGrid` - a regular rectangular grid.
Attempts to generate a symmetric positive semidefinite nested circulant matrix in which is embedded the covariance matrix corresponding to the given covariance function and grid. Using a binary search, it finds the minimum amount of padding, under the assumption of equal padding in each dimension, for the result to be positive semidefinite. The function returns both the matrix and the optimal padding.

If `covfun` is a [`HomogeneousCovFun`](@ref), the matrix is a [`NestedCirculant`](@ref).
If `covfun` is an [`IsotropicCovFun`](@ref), the matrix is a [`NestedSymmetricCirculant`](@ref). 
The returned padding is of type `NTuple{ndims(grid),Int}`

The optional arguments are the following:
  * `tol` - Due to numerical errors, it may happen that no amount of padding makes the matrix numerically positive semidefinite. Therefore, a matrix is regarded as positive definite if none of its eigenvalues are numerically below `-tol`.
  * `maxi` - The maximum allowed padding. If the smallest eigenvalue is not above `-tol`, will return that maximum with a warning.
  * `approx` - If `true`, finds the optimal padding (2ⁱ,…,2ⁱ) for natural i instead of the optimal padding (i,…,i). Should the optimal be larger than maxi, returns maxi. 
  * `print` - `false` for no printing, `true` for printing
"""
function circulantembed(covfun::Union{HomogeneousCovFun,IsotropicCovFun}, grid::RegularGrid; tol=1e-13, maxi=10000, approx=false, print=0)

    # Constructs the circulant matrix and padding values (i,…,i).
    function CE(i::Int)
        padding=ntuple(x->i,ndims(grid))
        C = circulantembed(covfun,grid,padding)
        return C,padding
    end

    # Check whether the padding (i,…,i) results in a positive semidefinite matrix and prints information if requested.
    function check_posdef(i::Int)
        C, padding = CE(i)
        posdef, minλ = isposdef_tol(C, tol)
        print>0 && println("With padding $i, smallest λ is $minλ.")
        return posdef
    end

    check_posdef(0) && return CE(0) # Trivial case; no padding required.

    # Location of minimum padding is narrowed down to the interval i/2+1:i
    i = 1
    while true
        check_posdef(i) && break
        i = i*2
        if i > maxi
            C, padding = CE(maxi)
            posdef, minλ = isposdef_tol(C,tol)
            if !posdef
                @warn("Padding requirement exceeds $maxi (smallest λ is $minλ). Returning CE with $maxi padding. Set negative eigenvalues to 0 to generate approximate stochastic field samples.")
                return C, padding
            end
            break
        end
    end
    approx && return CE(min(i,maxi)) # return approximate location

    # The minimum number of padding is now in i/2+1:i
    # A binary search in i/2+1:i now refines i
    a = floor(Int,i/2)+1
    b = min(i,maxi)
    while true
        a==b && break
        i = floor(Int,(a+b)/2)
        check_posdef(i) ? b = i : a = i + 1
    end
    check_posdef(a) && return CE(a)
end

# slower naive implementation, for testing purposes.
function circulantembed_naive(covfun::Union{HomogeneousCovFun,IsotropicCovFun}, grid::RegularGrid;tol=1e-13, maxi=250, print=0)
    local C, padding
    for i=0:maxi
        padding=ntuple(x->i,ndims(grid))
        C = circulantembed(covfun,grid,padding)
        posdef, minλ = isposdef_tol(C, tol)
        print>0 && println("With padding $i, smallest λ is $minλ.")
        posdef && (return C,padding)
    end
    minλ = minimum(real.(eigvals(C)))
    @warn("Padding requirement exceeds $maxi (smallest λ is $minλ). Returning CE with $maxi padding. Set negative eigenvalues to 0 to generate approximate stochastic field samples.")
    return C,padding
end

"""
    upstate!(state::Vector{Int},ends)->Bool
    
Auxiliary function that can be used for implementing the variable depth for-loop
    
    for state[end] = 1:ends[end]
        for state[end-1] = 1:ends[end-1]
            ...
                for state[1] = 1:ends[1]
                    #body 
                end 
            ...
        end
    end

This is done by

    # ends is assumed to be defined
    state = ones(Int,length(ends)) #loop state
    for i = 1:prod(ends)
        #body
        upstate!(state, ends)
    end    

Importantly, the for-loop depth `length(ends)` is allowed to be unknown until runtime.

upstate!(state,ends) traverses the elements in `state`. For each element, if it is smaller than the corresponding value in `ends`, that element is incremented and `false` is returned. If it is equal or larger than the corresponding element in `ends`, it is set to `1`, indicating that the for-loop corresponding to that element must now begin from the start. If all elements in `state` are larger than the corresponding elements in `ends`, the nested for-loops have finished and `true` is returned. The returned boolean can be used to implement the above with a while-loop instead.
"""
function upstate!(state::Vector{Int},ends)
    for d = 1:length(state)
        if state[d]≥ends[d]
            state[d]=1
        else
            state[d]+=1
            return false
        end
    end
    return true
end

"""
    circulantembed(covfun::IsotropicCovFun, grid, padding)->NestedSymmetricCirculant

  * `covfun::IsotropicCovFun` - An isotropic covariance function.
  * `grid::RegularGrid` - a regular rectangular grid with `N` dimensions, i.e., `ndims(grid)==N`.
  * `padding::NTuple{N,Int}` - the number of padding in each of the `N` dimensions.
Returns an `NestedSymmetricCirculant` in which the covariance matrix is embedded. The resulting matrix is not guaranteed to be positive definite. The `NestedSymmetricCirculant` is efficient and not stored explicitly, see [`NestedSymmetricCirculant`](@ref).
"""
function circulantembed(covfun::IsotropicCovFun, grid::RegularGrid, padding)
    ndims(grid)==length(padding) || error("The number of elements in the given padding vector must coincide with the number of dimensions in the grid.")

    Esize = Tuple(size(grid).+padding) # Extended grid size
    P = Array{Float64,ndims(grid)}(undef,Esize) # Allocate tensor that will store all distinct values of the covariance matrix
    Δs = getΔ(grid) # Get grid lengths in all directions

    state = ones(Int,ndims(grid)) #loop state
    for i = 1:prod(Esize)
        inds = state.-1
        point = Point(inds.*Δs)
        P[state...] = covfun(point)
        upstate!(state, Esize)
    end
    return NestedSymmetricCirculant(P)
end

"""
    circulantembed(covfun::HomogeneousCovFun, grid, padding)->NestedCirculant

  * `covfun::HomogeneousCovFun` - A homogeneous covariance function.
  * `grid::RegularGrid` - a regular rectangular grid with `N` dimensions, i.e., `ndims(grid)==N`.
  * `padding::NTuple{N,Int}` - the number of padding in each of the `N` dimensions.
Returns an `NestedCirculant` in which the covariance matrix is embedded. The resulting matrix is not guaranteed to be positive definite. The `NestedCirculant` is efficient and not stored explicitly, see [`NestedCirculant`](@ref).
"""
function circulantembed(covfun::HomogeneousCovFun, grid::RegularGrid, padding)
    ndims(grid)==length(padding) || error("The number of elements in the given padding vector must coincide with the number of dimensions in the grid.")

    Esize = Tuple(size(grid).+padding) # Extended grid size
    Psize = 2 .*Esize.-1 # Number of in general unique values in the covariance matrix
    P = Array{Float64,ndims(grid)}(undef,Psize) # Allocate tensor that will store all distinct values of the covariance matrix
    Δs = getΔ(grid) # Get grid lengths in all directions

    # This function constructs the index in the grid (having size Esize) of the Point that is relevant for the value at location `state` in the Array P (having size Psize). See code further below for more details. 
    function getinds(state)
        inds = Vector{Int}(undef,length(state))
        for d = 1:length(inds)
            inds[d] = state[d]<=Esize[d] ? state[d]-1 : state[d]-2*Esize[d]
        end
        return inds
    end

    state = ones(Int,ndims(grid)) #loop state
    for i = 1:prod(Psize)
        inds = getinds(state) # Construct indices of the relevant.
        point = Point(inds.*Δs) # Get/Construct relevant point.
        P[state...] = covfun(point) # Evaluate the covariance function.
        upstate!(state, Psize) # update the loop variables.
    end
    return NestedCirculant(P)
end

#################
### AUXILIARY ###
#################
"""
    symmetrize(A::AbstractArray{T,N}) where {T,N}

Symmetrizes A in all dimensions. E.g., `symmetrize([a₁,…,aₙ])` yields `[a₁,…,aₙ,aₙ₋₁,…,a₂]`. Each dimension size `nᵢ` will become `2nᵢ-2`.
"""
function symmetrize(A::AbstractArray{T,N}) where {T,N}
    dims = 2collect(size(A)).-2
    B = zeros(T,dims...)
    # iterate over all corners
    for corner = 0:2^N-1
        bs = Bool.(digits(corner,base=2,pad=N)) # bitrepresentation of corner number
        # build ranges for that corner
        rangesA = collect(bs[i] ? (2:size(A)[i]-1) : (1:size(A)[i]) for i in 1:N)
        rangesB = collect(bs[i] ? (size(B)[i]:-1:size(A)[i]+1) : (1:size(A)[i]) for i in 1:N)
        B[rangesB...] = A[rangesA...]
    end
    return B
end
# NOTE: A different symmetrize can be defined such that `symmetrize2([a₁,…,aₙ])` yields `[a₁,…,aₙ,aₙ,…,a₂]`. If then `NestedSymmetricCirculant2(A) == NestedCirculant(symmetrize2(A))`, the padding requirements to make this matrix positive semidefinite would be lower. The only difference for the calculation of eigenvalues would be to employ a different type of fft that implicitly symmetrizes as `symmetrize2` instead of `symmetrize`. This would be a discrete cosine transform of a different type instead of type 1 (see fct information below).

# does padding for the matrix with zeros. The size of A is increased by n
function pad(A::AbstractArray{T,N}, n::Vararg{Int,N}) where {T,N}
    newsize = size(A).+n
    B = zeros(T,newsize...)
    B[axes(A)...] = A;
    return B
end
pad(A::AbstractArray{T,N}, n::Int) where {T,N} = pad(A, fill(n,N)...)

# AUX: Fast implementation of the real to real discrete cosine transform of type 1
# Note: for real symmetric data v = [v₀, v₁, …, vₙ, vₙ₋₁, …, v₁], we have
# y = fft(v) = [y₀, y₁, …, yₙ, yₙ₋₁, …, y₁] is real and symmetric.
# This real arithmetic function fct can be used to speed up this computation since
# fct([v₀, v₁, …, vₙ]) = [y₀, y₁, …, yₙ].
# Call symmetrize(fct(v)) to reproduce fft(v) if needed.
fct(v::Array{<:Real},dims) = FFTW.r2r(v,FFTW.REDFT00,dims)
fct(v::Array{<:Real}) = FFTW.r2r(v,FFTW.REDFT00)

function rfft_to_fft(A::Array{<:Number},even::Bool)
    d = ndims(A)
    inv = [size(A,i):-1:1 for i in 2:d]
    Amirror = A[end-even:-1:2,inv...]
    return cat(A,conj.(Amirror);dims=1)
end

########################
### NESTED CIRCULANT ###
########################
"""
    ANestedCirculant <: AbstractMatrix{E}

Abstract nested circulant matrix. 

See also [`NestedCirculant`](@ref) and [`NestedSymmetricCirculant`](@ref).
"""
abstract type ANestedCirculant{E} <: AbstractMatrix{E} end

"""
    NestedCirculant{E} <: ANestedCirculant{E} <: AbstractMatrix{E}

A nested circulant matrix, i.e., a blockcirculant matrix with each block being blockcirculant with each block ... with each block being circulant. 

As an example, a 4×4 circulant matrix is of the form 

    [a1  a4  a3  a2
     a2  a1  a4  a3
     a3  a2  a1  a4
     a4  a3  a2  a1]

which can be generated by `NestedCirculant([a1, a2, a3, a4])`. The first column corresponds to the given column vector. A blockcirculant matrix with circulant blocks is of the form

    [A1  A4  A3  A2
     A2  A1  A4  A3
     A3  A2  A1  A4  
     A4  A3  A2  A1]

where each of the blocks is circulant. If `A1 = NestedCirculant([a11, a21, a31, a41])`, …, `A4 = NestedCirculant([a14, a24, a34, a44])`, then the above matrix can be generated by 

    A = [a11 a12 a13 a14
         a21 a22 a23 a24
         a31 a32 a33 a34
         a41 a42 a43 a44]
    NestedCirculant(A)

Thus, block `A1` is defined by the first column of `A`, etc. `A` can be generalized to a tensor of arbitrary order to generate arbitrarily deeply nested blockcirculant matrices. Note that the NestedCirculant matrix is stored implicitly and efficiently by storing only the given `A`.

See also [`NestedSymmetricCirculant`](@ref).
"""
struct NestedCirculant{E} <: ANestedCirculant{E}
    # tensor containing data. Each column (variable first index) corresponds to a
    # circulant matrix.
    # Each matrix (variable first and second index) corresponds
    # to a block circulant matrix with each block circulant determined by the columns
    # This easily generalizes to an arbitrary number of dimensions.
    A::AbstractArray{E}
end

"""
    NestedSymmetricCirculant{E} <: ANestedCirculant{E} <: AbstractMatrix{E}

A nested symmetric circulant matrix, i.e., a symmetric blockcirculant matrix with each block being symmetric blockcirculant with each block ... with each block being symmetric circulant. Note that `NestedSymmetricCirculant(A) == NestedCirculant(symmetrize(A))`. 

As an example, a 4×4 symmetric circulant matrix is of the form 

    [a1  a2  a3  a2
     a2  a1  a2  a3
     a3  a2  a1  a2
     a2  a3  a2  a1]

which can be generated by `NestedCirculant([a1, a2, a3])`. The first column corresponds to `symmetrize([a1, a2, a3])`. A symmetric blockcirculant matrix with circulant blocks is of the form

    [A1  A2  A3  A2
     A2  A1  A2  A3
     A3  A2  A1  A2  
     A2  A3  A2  A1]

where each of the blocks is symmetric circulant. If `A1 = NestedSymmetricCirculant([a11, a21, a31])`, …, `A3 = NestedSymmetricCirculant([a13, a23, a33])`, then the above matrix can be generated by 

    A = [a11 a12 a13
         a21 a22 a23
         a31 a32 a33]
    NestedSymmetricCirculant(A)

Thus, block `A1` is defined by the first column of `A`, etc. `A` can be generalized to a tensor of arbitrary order to generate arbitrarily deeply nested symmetric blockcirculant matrices. Note that the NestedSymmetricCirculant matrix is stored implicitly and efficiently by storing only the given `A`.

See also [`NestedCirculant`](@ref).
"""
struct NestedSymmetricCirculant{E} <: ANestedCirculant{E}
    # tensor containing data. Each column (variable first index) corresponds to a
    # circulant matrix. Each matrix (variable first and second index) corresponds
    # to a block circulant matrix with each block circulant determined by the columns
    # This easily generalizes to an arbitrary number of dimensions.
    A::AbstractArray{E}
end

# matrix functions
eltype(C::ANestedCirculant{E}) where E = E
ndims(C::ANestedCirculant) = 2;
size(C::NestedCirculant, n::Int) = length(C.A)
size(C::NestedSymmetricCirculant, n::Int) = prod(2 .*size(C.A).-2)
size(C::ANestedCirculant) = (size(C,1),size(C,2))
length(C::ANestedCirculant) = prod(size(C))
axes(C::ANestedCirculant, n::Int) = Base.OneTo(size(C,n))
axes(C::ANestedCirculant) = (axes(C,1),axes(C,2))

"""
    getblockstructure(C::ANestedCirculant)

returns `(s₁, …, sₙ)` where n is equal to the number of times the blocks are nested. For an unnested matrix, n=1. For a matrix consisting of blocks of elements, n=2, etc. The given `C` then contains sₙ blocks in each row and each column, each of which contain sₙ₋₁ blocks in each row and each column, ..., each of which contain s₁ elements in each row and each column. The basic (inner) blocks thus contain s₁×s₁ elements. We then have `size(C,1) == size(C,2) == prod(getblockstructure(C))`.
"""
getblockstructure(C::NestedCirculant) = size(C.A)
getblockstructure(C::NestedSymmetricCirculant) = 2 .*size(C.A).-2

#returns (block number,remainder index)
#assumes the matrix is partitioned in m blocks of size p×p
function getblocknumber(i,j,m,p)
    q1,r1 = divrem(i-1,p)
    q2,r2 = divrem(j-1,p)
    mod(q1-q2,m)+1,r1+1,r2+1
end
function getindex(C::NestedCirculant,i::Int,j::Int)
    s = size(C.A)
    p = prod(s)
    loc = Vector{Int}(undef,ndims(C.A))
    for d in ndims(C.A):-1:1
        p /= s[d]
        loc[d],i,j = getblocknumber(i,j,s[d],p)
    end
    C.A[loc...]
end
function getindex(C::NestedSymmetricCirculant,i::Int,j::Int)
    s = 2 .*size(C.A).-2 #s contains size of symmetrize(C.A)
    p = prod(s)
    loc = Vector{Int}(undef,ndims(C.A))
    for d in ndims(C.A):-1:1
        p /= s[d]
        loc[d],i,j = getblocknumber(i,j,s[d],p)
        if loc[d] > size(C.A,d); loc[d] = s[d]-loc[d]+2 end
    end
    C.A[loc...]
end
#setindex!(C::Circulant,i::Int,j::Int) = @error("setting index for a circulant matrix is not supported.")

# arithmetic NestedCirculant
+(X::NestedCirculant, Y::NestedCirculant) = NestedCirculant(X.A+Y.A)
-(X::NestedCirculant, Y::NestedCirculant) = NestedCirculant(X.A-Y.A)
-(X::NestedCirculant) = NestedCirculant(-X.A)
*(X::NestedCirculant, c::Number) = NestedCirculant(X.A*c)
*(c::Number, X::NestedCirculant) = *(X,c) #only right vector multiplication implemented
function *(X::NestedCirculant, V::Vector{<:Number})
    length(V) == size(X,2) || (error("dimension mismatch"); return)
    s = size(V)
    T = fft(X.A).*fft(reshape(V,size(X.A)))
    reshape(real(ifft!(T)),s)
end

# arithmetic NestedSymmetricCirculant
+(X::NestedSymmetricCirculant, Y::NestedSymmetricCirculant) = NestedSymmetricCirculant(X.A+Y.A)
-(X::NestedSymmetricCirculant, Y::NestedSymmetricCirculant) = NestedSymmetricCirculant(X.A-Y.A)
-(X::NestedSymmetricCirculant) = NestedSymmetricCirculant(-X.A)
*(X::NestedSymmetricCirculant, c::Number) = NestedSymmetricCirculant(X.A*c)
*(c::Number, X::NestedSymmetricCirculant) = *(X,c)
function *(X::NestedSymmetricCirculant, V::Vector{<:Number}) #only right vector multiplication implemented
    length(V) == size(X,2) || (error("dimension mismatch"); return)
    s = size(V)
    T = symmetrize(fct(X.A)).*fft(reshape(V,2 .*size(X.A).-2))
    reshape(real(ifft!(T)),s)
end

# spectrum
eigen(X::ANestedCirculant) = Eigen(eigvals(X),eigvecs(X))

"""
    isposdef_tol(X::ANestedCirculant, tol_real=1e-13, tol_imag=1e-10)

Returns whether the smallest eigenvalue is larger than `-tol` and returns the real part of that smallest eigenvalue. Also checks whether the imaginary part of all eigenvalues is smaller in absolute value than `tol_imag`. 
"""
function isposdef_tol(X::ANestedCirculant, tol_real=1e-13, tol_imag=1e-10)
    λs = eigvals(X)
    minλ = minimum(real.(λs))
    return (all(abs.(imag.(λs)).<=tol_imag) && all(real.(λs).>=-tol_real)), minλ
end

#NOTE: For NestedSymmetricCirculant, the eigenvectors can all be made real (by linear combinations, e.g., summing/taking real and imag parts since they are conjugated anyway)
function eigvecs(X::ANestedCirculant{E}) where E
    es = Matrix{Complex{Float64}}(undef,size(X))
    for i = 1:size(X,1)
        es[:,i] = eigvec(X,i)
    end
    return es
end

"""
    eigvec(C::ANestedCirculant{E}, i::Int)

Returns the `i`-th eigenvector of `C`. 
"""
function eigvec(X::ANestedCirculant{E}, i::Int) where E
    m = zeros(E, getblockstructure(X));
    m[i] = 1
    return reshape(ifft(m),size(X,1))
end
eigvals(X::NestedCirculant) = (fft(X.A))[:]
eigvals(X::NestedSymmetricCirculant) = (symmetrize(fct(X.A)))[:]

#########################
### SAMPLER GENERATOR ###
#########################
"""
    gen_sampler(sf::StochField, grid::RegularGrid; seed=rand(UInt), extendsamples::Bool=false, CEoptions...)->sampler::Function

Generates a sampler to sample the stochastic field `sf` on the grid `grid` using the circulant embedding (CE) method for Gaussian and Lognormal stochastic fields. For the other trivial stochastic field types, trivial methods are used; see [`StochField`](@ref). 

Returns a function `sampler` with signature 

    sampler(i::Int) -> Array{Float64, ndims(grid)}

which is used to generate the `i`-th sample. The samples are reproducible since sampler(i) always returns the same result. The optional `seed` argument of `gen_sampler` allows for the creation of different independent samplers.  

The CE method first attempts to embed the covariance matrix in a circulant matrix using `circulantembed(sf.covfun,grid; CEoptions)`, i.e., searching an appropriate amount of grid padding such that the resulting circulant matrix is positive semidefinite. This part can be tuned using `CEoptions`. The sampler returns samples on the full extended grid if `extendsamples=true` is passed. A reference to the grid can be found using `sampler.extendedgrid`. 

# Basic usage
```julia-repl
sf = Gaussian(exponentialcovariance(0.3,0.5))
grid = RegularGrid(33,33)
sampler = gen_sampler(sf, grid)
sam(1)
```
"""
function gen_sampler(sf::Gaussian, grid_given::RegularGrid; seed=rand(UInt), extendsamples::Bool=false, CEoptions...)
    C,padding = circulantembed(sf.covfun,grid_given; CEoptions...)

    getλs(C::NestedCirculant) = real.(fft(C.A))
    getλs(C::NestedSymmetricCirculant) = symmetrize(fct(C.A))
    λs = max.(0.0,getλs(C)) # All eigenvalues should be real nonnegative numbers. Here we ensure this, should it not be the case due to numerical errors or insufficient padding.
    
    c = 1/√size(C,1)
    sqrtλs=c.*sqrt.(λs)
    transform = plan_rfft(sqrtλs; flags=FFTW.MEASURE)
    seed_derived = Base.hash(seed)

    if extendsamples
        extendedgrid = extend(grid_given,size(λs).-size(grid_given)) #full grid corresponding to the circulant matrix
        extendedmean = sf.mean.(extendedgrid)
        function extendedsampler(i::Int) # function generates sample on the grid it was constructed at.
            extendedgrid
            Random.seed!(seed_derived+i) # initializes random number generator with seed equal to seed_derived+i
            extendedmean.+extendedsample_(sqrtλs; F=transform)
        end
        return extendedsampler
    else
        grid = grid_given
        mean = sf.mean.(grid)
        function sampler(i::Int) # function generates sample on the grid it was constructed at.
            grid
            Random.seed!(seed_derived+i) # initializes random number generator with seed equal to seed_derived+i
            mean.+sample_(sqrtλs; F=transform)[axes(grid)...]
        end
        return sampler
    end
end

function gen_sampler(seed, sf::LogNormal, grid::RegularGrid; CEopts...)
    gaussian_sampler = gen_sampler(seed,Gaussian(sf.mean,sf.covfun),grid; CEopts...)
    return i::Int->exp.(gaussian_sampler(i))
end

function gen_sampler(seed, sf::Uniform, grid::RegularGrid)
    seed_derived = Base.hash(seed)
    function sampler(i::Int)
        Random.seed!(seed_derived+i)
        value = sf.lowest + rand()*(sf.highest-covfun.lowest)
        fill(value,size(grid))
    end
end

function gen_sampler(seed, sf::Deterministic, grid::RegularGrid)
    deterministic_values = sf.fun.(grid)
    sampler(i::Int) = deterministic_values
end

# internal function for efficient sampling given sqrtλs
# assuming:
#   all(size(sqrtλs).==size(points))
#   all(sqrtλs.>=0.0)
function sample_(sqrtλs::Array{Float64,N}, noise::Array{Float64,N}=randn(size(sqrtλs)); F=plan_rfft(sqrtλs)) where N
    # F contains (part of) the fft matrix (implicitly)
    complexdata = F*(sqrtλs.*noise)
    data = real.(complexdata).+imag.(complexdata) # Discrete Hartley transform
    data::Array{Float64,N}
end

# internal function for efficient sampling given sqrtλs
# assuming:
#   all(size(sqrtλs).==size(points))
#   all(sqrtλs.>=0.0)
function extendedsample_(sqrtλs::Array{Float64,N}, noise::Array{Float64,N}=randn(size(sqrtλs)); F=plan_rfft(sqrtλs)) where N
    # F contains (part of) the fft matrix (implicitly)
    complexdata = F*(sqrtλs.*noise)
    complexdata = rfft_to_fft(complexdata,iseven(size(sqrtλs,1)))
    data = real.(complexdata).+imag.(complexdata) # Discrete Hartley transform
    data::Array{Float64,N}
end

end
