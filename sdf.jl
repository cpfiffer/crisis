# simulates a stochastic discount factor
using Distributions
using LinearAlgebra
using StatsPlots
using DataFrames

n_assets = 100
n_states = 2

pi = [0.5, 0.5]
A = [
    0.90 0.10;
    0.01 0.99
]

# Stochastic parameters
iw = InverseWishart(n_assets + 2, diagm(ones(n_assets)))
mu = [randn(n_assets) .+ 1 for _ in 1:n_states]
sigma = [rand(iw) for _ in 1:n_states]

# Deterministic parameters
# mu =[
#     [1.5, 2.2],
#     [1.6, 1.9],
# ] ./ 5
# sigma = [
#     [2.0 0.25; 0.25 3.0],
#     [3.0 -0.22; -0.22 3.1],
# ]

N = 100_000
gamma = 0
w = repeat([1/n_assets], n_assets)
delta = 0.9

function hmm_states(A, T, pi)
    states = zeros(Int, T)
    states[1] = rand(Categorical(pi))
    for t in 2:T
        states[t] = rand(Categorical(A[states[t-1], :]))
    end

    return states
end

function hmm_emit(states, mus, sigmas)
    f = zeros(length(states), length(mus[1]))
    dists = [MvNormal(mus[i], sigmas[i]) for i in 1:length(mus)]

    for i in 1:length(states)
        f[i, :] = rand(dists[states[i]])
    end

    return f
end

function hmm_payoffs(mu, sigma, N, A, state)
    states = rand(Categorical(A[state, :]), N)
    draws = zeros(n_assets, N)
    
    for i in 1:N
        s = states[i]
        draws[:, i] = rand(MvNormal(mu[s], sigma[s]))
    end

    return draws
end

function conditional_payoffs(mus, sigmas, N, probs)
    dists = [MvNormal(mus[i], sigmas[i]) for i in 1:length(mus)]
    s = rand(Categorical(probs), N)
    draws = zeros(length(mus[1]), N)

    for i in 1:N
        draws[:,i] = rand(dists[s[i]])
    end

    return draws
end

function payoffs(mu, sigma, N)
    dist = MvNormal(mu, sigma)
    draws = rand(dist, N)
    return draws
end

function prices(delta, w, gamma, draws)
    m = w'exp.(draws) .^ (-gamma)
    return vec(delta .* mean(m .* draws, dims=2))
end

function emit_density(f, dist)
    return exp(loglikelihood(dist, f))
end

function filtering(f, A, mus, sigmas, pi)
    K = size(A, 1)
    T = size(f, 1)
    N = size(f, 2)

    # Generate distributions
    dists = [MvNormal(mus[i], sigmas[i]) for i in 1:K]

    # Preallocate likelihoods
    α = zeros(K, T)
    predicted = zeros(K, T)
    α[:,1] = pi .* map(i -> emit_density(f[1,:], dists[i]), 1:K)
    α[:,1] = α[:,1] ./ sum(α[:,1])

    for k in 1:K
        predicted[1,1] += A[k,1] * α[k, 1]
        predicted[2,1] += A[k,2] * α[k, 1]
    end

    for t in 2:T
        α[1,t] = emit_density(f[t,:], dists[1])
        α[2,t] = emit_density(f[t,:], dists[2])

        for k in 1:K
            α[1,t] += A[k,1] * α[k, t-1]
            α[2,t] += A[k,2] * α[k, t-1]
        end

        α[:,t] = α[:,t] ./ sum(α[:,t])

        for k in 1:K
            predicted[1,t] += A[k,1] * α[k, t]
            predicted[2,t] += A[k,2] * α[k, t]
        end
    end

    return α, predicted
end

function conditional_prices(mu, sigma, N, probs, delta, w, gamma)
    p = zeros(size(mu[1], 1), size(probs, 2))
    for t in 1:size(probs, 2)
        draws = conditional_payoffs(mu, sigma, N, probs[:,t])
        p[:,t] = prices(delta, w, gamma, draws)
    end
    return p
end

S = hmm_states(A, 100, pi)
f = hmm_emit(S, mu, sigma)
α, predicted = filtering(f, A, mu, sigma, pi)

p_expected = conditional_prices(mu, sigma, 10_000, predicted, delta, w, gamma)
p_true = conditional_prices(mu, sigma, 10_000, α, delta, w, gamma)

R = p_expected[:, 2:end] ./ p_expected[:,1:end-1] .- 1

splot = plot(S)

price_plot = plot((p_expected')[:,1])

plot(price_plot, splot)
# plot!(S .- 1)

# draws1 = conditional_payoffs(mu, sigma, 10000, α[:,1])
# p1 = prices(delta, w, gamma, draws1)

# draws2 = conditional_payoffs(mu, sigma, 10000, predicted[:,1])
# p2 = prices(delta, w, gamma, draws2)

# df = DataFrame(
#     p1 = p1,
#     p2 = p2,
#     r = p2 ./ p1 .- 1,

# )

# display(α)
# display(predicted)

# draws_1 = payoffs(mu[1], sigma[1], N)
# draws_2 = payoffs(mu[2], sigma[2], N)

# prices_1 = prices(delta, w, gamma, draws_1)
# prices_2 = prices(delta, w, gamma, draws_2)

# hmm_1 = hmm_payoffs(mu, sigma, N, A, 1)
# hmm_2 = hmm_payoffs(mu, sigma, N, A, 2)

# prices_hmm_1 = prices(delta, w, gamma, hmm_1)
# prices_hmm_2 = prices(delta, w, gamma, hmm_2)

# df = DataFrame(
#     mu_1 = mu[1],
#     mu_2 = mu[2],
#     sigma_1 = diag(sigma[1]),
#     sigma_2 = diag(sigma[2]),
#     p_1 = vec(prices_1),
#     p_2 = vec(prices_2),
#     prices_hmm_1 = vec(prices_hmm_1),
#     prices_hmm_2 = vec(prices_hmm_2),
# )

# p = round.(
#     delta .* mean(m .* draws, dims=2),
#     digits=3
# )
