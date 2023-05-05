"""
    Points

Module for representing a point in n-dimensional space.

See [`Point`](@ref).
"""
module Points

export Point 

import Base: ==, +, -, *, /, \
import Base: show
import Base: getproperty, propertynames
import LinearAlgebra: norm

##############
### POINTS ###
##############
"""
    Point{N}

Representation of an `N` dimensional point. The values are stored as `N` `Float64`s. Points can be constructed as 

p = Point(x,y,z,...)
p = Point([x,y,z,...])

All coordinates are immutable and can be read using `p.coordinates`, returning an `NTuple{N,Float64}`. For individual coordinates there are the shorthands `p.x`, `p.y`, `p.z` if they exist.

Operations `==`, `+`, `-`, `*`, `/`, and `\\` and `norm` are defined and behave as expected.
"""
struct Point{N}
    coordinates::NTuple{N,Float64}
end
Point{N}(x...) where N = Point{N}(NTuple{N,Float64}(x))
Point(x...) = Point{length(x)}(x...)
Point(x) = Point{1}(tuple(x))
Point(x::AbstractVector) = Point{length(x)}(tuple(x...))

function getproperty(p::Point{N}, s::Symbol) where N
    if s == :x && N>=1
        return p.coordinates[1]
    elseif s == :y && N>=2
        return p.coordinates[2]
    elseif s == :z && N>=3
        return p.coordinates[3]
    else
        return getfield(p,s)
    end
end
propertynames(p::Point{N}) where N = (:coordinates,:x,:y,:z)[1:min(N+1,4)]

show(io::IO, p::Point) = print(io, "P$(p.coordinates)")
show(io::IO, p::Point{1}) = print(io, "P($(p.x))")
==(a::Point,b::Point) = a.coordinates == b.coordinates
+(a::Point, b::Point) = Point(a.coordinates .+ b.coordinates)
-(a::Point, b::Point) = Point(a.coordinates .- b.coordinates)
-(p::Point) = Point(.-p.coords)
*(p::Point, c::Number) = Point(p.coordinates .* c)
*(c::Number, p::Point) = *(p,c)
/(p::Point, c::Number) = Point(p.coordinates ./ c)
\(c::Number, p::Point) = /(p,c)
norm(a::Point, p::Real=2) = norm(collect(a.coordinates), p)

end