using Distributions
using Polynomials
using QuadGK
using HCubature

d = MvNormal(5, 1)
infs = [Inf for i in 1:size(d)[1]]
ninfs = [-Inf for i in 1:size(d)[1]]

hcubature(
    x -> pdf(d, x), 
    -ones(5) .* 90, 
    ones(5) .* 90, 
    maxevals=1000000000
)


hcubature(
    x -> logpdf(d, x), 
    ninfs, 
    infs, 
    maxevals=100000000,
    initdiv=100
)