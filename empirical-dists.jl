using Distributions
using StatsPlots
using Random

Random.seed!(5)
N = 5
μ = randn(N)
σ = randn(N) .^ 2
dists = [Normal(μ[i], σ[i]) for i in 1:length(μ)]

# function wealth(dists, w0)
#     rets = map(x -> 1 + rand(x), dists)
#     return w0 * prod(rets)
# end

# simulations = map(_ -> wealth(dists, 1), 1:10000000)

# true_mean = prod(1 .+ μ)
# true_var = prod([σ[i]^2 + (1 + μ[i])^2 for i in 1:length(μ)]) - prod((1 .+ μ).^2)
# @info "" mean(simulations) true_mean var(simulations) true_var

mu = [1.0, 2.2]
sig = [1.0 0.1; 0.1 2.0]
G = MvNormal(mu, sig)

function sim_mean(n, g, w)
    r = size(g)[1]
    draws = zeros(r + 1, n)
    draws[1:r, :] = rand(g, n)
    draws[end, :] = w'draws[1:r, :]

    return draws
end

w = [1/2, 1/2]
d = sim_mean(1000000, G, w)
guess_sig = cov(d')
mean_cov_1 = w'sig[1,:]
mean_cov_2 = w'sig[2,:]
true_sig = [
    sig[1,1] sig[1,2] mean_cov_1;
    sig[2,1] sig[2,2] mean_cov_2;
    mean_cov_1 mean_cov_2 w'sig*w
]