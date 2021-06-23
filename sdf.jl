# simulates a stochastic discount factor
using Distributions
using LinearAlgebra
using StatsPlots
using DataFrames

n_assets = 5
n_states = 2

pi = [0.5, 0.5]
A = [
    0.8 0.2;
    0.2 0.8
]

iw = InverseWishart(n_assets + 2, diagm(ones(n_assets)))
mu = [randn(n_assets) .+ 1 for _ in 1:n_states]
sigma = [rand(iw) ./ 3 for _ in 1:n_states]

N = 100_000
gamma = 2
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

function payoffs(mu, sigma, N)
    dist = MvNormal(mu, sigma)
    draws = rand(dist, N)
    return draws
end

function prices(delta, w, gamma, draws)
    m = w'exp.(draws) .^ (-gamma)
    return delta .* mean(m .* draws, dims=2)
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
    b = map(x -> (u -> pdf(x, u)), dists)

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

S = hmm_states(A, 100, pi)
f = hmm_emit(S, mu, sigma)
α, predicted = filtering(f, A, mu, sigma, pi)

display(α)
display(predicted)

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
