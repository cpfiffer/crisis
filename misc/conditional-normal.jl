import Pkg; Pkg.activate(".")

using Distributions
using LinearAlgebra
using StatsPlots

N = 10_000
μ = 15 .* randn(N)
Σ = 15 .* rand(InverseWishart(N + 1, 0.1 .* diagm(ones(N))))
dist = MvNormal(μ, Σ)
data = rand(dist)

function partition(M, n)
    M_a = M[1:n, 1:n]
    M_ab = M[n+1:end, 1:n]
    M_ba = M[1:n, n+1:end]
    M_b = M[n+1:end, n+1:end]

    return (M_a, M_ab, M_ba, M_b)
end

function conditional(X, mu, sigma, n)
    x_a = X[1:n]
    mu_a = mu[1:n]
    mu_b = mu[n+1:end]
    (M_a, M_ab, M_ba, M_b) = partition(sigma, n)


    μ_bar = mu_b + M_ab * inv(M_a) * (x_a - mu_a)
    Σ_bar = Symmetric(M_b - M_ab * inv(M_a) * M_ba)

    # return μ_bar, Σ_bar
    return MvNormal(μ_bar, Σ_bar)
end

function samples(X, mu, sigma, n, n_samples=10_000)
    cdist = conditional(X, mu, sigma, n)

    s = rand(cdist, n_samples)

end

# samples()
ss = []
for n in 1:(div(N, 5)):N-1
    s = samples(data, μ, Σ, n)
    push!(ss, s'[:,end])
    # sleep(0.1)
    # density(s'[:,end]) |> display
end

density(hcat(ss...))
