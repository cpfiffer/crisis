using Distributions
using Polynomials
using QuadGK

function kl(mutual, pairwise::Vector{<:Distribution})
    fns = [x -> pdf(p, x) for p in pairwise]
    return kl(mutual, fns)
end

function kl(mutual::Distribution, pairwise::Vector{<:Function})
    g(x...) = pdf(mutual, [m for m in x])
    f(x...) = prod([p(y) for (p,y) in zip(pairwise, x)])

    function target(x, y)
        v = g(x,y) * (log(g(x,y)) - log(f(x,y)))
        if isnan(v) || !isfinite(v)
            return 0
        else
            return v
        end
    end

    t(z) = target(z[1], z[2])

    return quadgk(x -> quadgk(m -> t([x, m]), -Inf, Inf)[1], -Inf, Inf)[1]
    # return quadgk(x -> quadgk(m -> t([x, m]), -Inf, Inf), -Inf, Inf)
    # return target(1.0, 2.0)
end

function kl(mutual::Distribution, pairwise::Distribution)
    g(x...) = pdf(mutual, [m for m in x])
    f(x...) = pdf(pairwise, [m for m in x])
   
    function target(x, y)
        v = g(x,y) * (log(g(x,y)) - log(f(x,y)))
        if isnan(v) || !isfinite(v)
            return 0
        else
            return v
        end
    end

    t(z) = target(z[1], z[2])

    return quadgk(x -> quadgk(m -> t([x, m]), -Inf, Inf)[1], -Inf, Inf)[1]
end

function mixture_entropy(m)
    return -quadgk(x -> pdf(m,x) * logpdf(m, x), -Inf, Inf)[1]
end

d1 = Normal(0, 0.5)
d2 = Normal(0, 1)

# mn1 = MvNormal(
#     [0,0],
#     [0.5 0.25; 0.25 0.5]
# )

# mn2 = MvNormal(
#     [0,0],
#     [0.5 0; 0 0.5]
# )

m1 = MixtureModel([
    Normal(0, 0.3),
    Normal(1, 0.2)
])

m2 = MixtureModel([
    Normal(0, 0.2),
    Normal(1, 0.3)
])

@info "" mixture_entropy(m1) mixture_entropy(m2)
# mixture_entropy(mn3)

# kl(mn1, [d1,d2])
# @info "correlated" kl(mn1, [d1,d2])
# @info "uncorrelated" kl(mn2, [d1,d2])
# @info "mixture" kl(mn3, mn2)