"""
    RegularGrids

Module for representing regular rectangular grids of point of any dimension, see [`RegularGrid`](@ref).
"""
module RegularGrids

using Points

export Point
export RegularGrid
export getΔ
export extend

import Base: ==
import Base: show
import Base: getindex, size, ndims, eltype

#####################
### Regular Grids ###
#####################
"""
    RegularGrid{N} <: AbstractArray{Point{N},N}

Regular rectangular grid, also known as uniform rectilinear grid, with `N` dimensions.

See also [`Point`](@ref), [`getΔs`](@ref), [`extend`](@ref).
"""
struct RegularGrid{N} <: AbstractArray{Point{N},N}
    ranges::NTuple{N,LinRange{Float64,Int}}
end

RegularGrid{N}(m::Vararg{Int,N}; start=zeros(N), stop=ones(N)) where N = RegularGrid{N}(ntuple(i->LinRange(start[i],stop[i],m[i]),N))

"""
    RegularGrid(m... [, start=start, stop=stop])

Constructs a `RegularGrid{N}` with `N = length(m)` with `m[i]` giving the number of points in dimension `i`. The grid spans from 0.0 to 1.0 in each dimension, unless specified by the arguments start and stop, in which case the grid spans from `start[i]` to `stop[i]` in dimension `i` instead.
"""
RegularGrid(m...; start=zeros(length(m)), stop=ones(length(m))) = RegularGrid{length(m)}(m...; start=start, stop=stop)

# ### Basic operations
_getΔ(nodes) = (nodes[end]-nodes[1])/(length(nodes)-1)

"""
    getΔ(g::RegularGrid{N}) -> NTuple{Float64,N}

Returns the grid spacings for all `N` dimensions.
"""
getΔ(g::RegularGrid) = _getΔ.(g.ranges)

"""
    getΔ(g::RegularGrid, i::Int) -> Float64

Returns the grid spacing for dimension `i`.
"""
getΔ(g::RegularGrid, i::Int) = _getΔ(g.ranges[i])
ndims(g::RegularGrid{N}) where N = N
eltype(::Type{RegularGrid{N}}) where N = Point{N}
size(g::RegularGrid) = Tuple(length.(g.ranges))
getindex(m::RegularGrid{N}, is::Vararg{Int,N}) where N = Point([m.ranges[d][is[d]] for d=1:N]...)
==(g1::RegularGrid,g2::RegularGrid) = g1≡g2 || all(g1.ranges==g2.ranges) # more efficient than checking all points separately 

### Printing
function show(io::IO, m::MIME"text/plain", grid::RegularGrid{N}) where N
    s = prod("$(grid.ranges[i].len)×" for i = 1:N-1)*"$(grid.ranges[end].len) $(typeof(grid)) for "
    s *= prod("[$(grid.ranges[i].start),$(grid.ranges[i].stop)]×" for i = 1:N-1)*"[$(grid.ranges[end].start),$(grid.ranges[end].stop)]"
    print(io, s)
end
show(io::IO, m::Type{MIME"text/plain"}, grid::RegularGrid{N}) where N = print(prod("$(grid.ranges[i].len)×" for i = 1:N-1)*"$(grid.ranges[end].len) $(typeof(grid))")
show(io::IO, grid::RegularGrid) = show(io, MIME"text/plain", grid)

### extending, i.e., padding
"""
    extend(g::RegularGrid{N}, extra::NTuple{N,Int}) -> RegularGrid{N}

Returns a new grid with the same grid spacings but extended with `extra[i]` points in dimension `i`.
"""
function extend(g::RegularGrid{N}, extra::NTuple{N,Int}) where N
    Δs = getΔ(g)
    m = length.(g.ranges) .+ extra
    ranges = ntuple(d->LinRange(g.ranges[d].start,g.ranges[d].stop+extra[d]*Δs[d],m[d]), N)
    RegularGrid{N}(ranges)
end

"""
    extend(g::RegularGrid{N}, extra::Int) -> RegularGrid{N}

Returns a new grid with the same grid spacings but extended with `extra` points in each of the dimensions.
"""
extend(m::RegularGrid, extra::Int) = extend(m,ntuple(x->extra,ndims(m)))

end
