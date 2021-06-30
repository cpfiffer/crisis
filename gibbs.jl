using Distributions: maximum
using Distributions
using LinearAlgebra
using Statistics
using Parameters
using StatsPlots
using Random
using StructArrays
using ProgressMeter
using StatsFuns
using DataFrames

function rescale_lp(lp)
    weights = exp.(lp .- maximum(lp))
    return weights ./ sum(weights)
end


function posterior_mu(m0, V0, Σ, X)
    μ = vec(mean(X; dims=1))
    N = size(X, 1)

    if N > 0
        V0_inv = inv(V0)
        Σ_inv = inv(Σ)
        VN = Symmetric(inv(V0_inv + N .* Σ_inv))
        mN = VN * (Σ_inv * (N .* μ)  + V0_inv * m0)
        return MvNormal(mN, VN)
    else
        return MvNormal(m0, V0)
    end
end

function posterior_variance(ν0, S0, μ, X)
    N = size(X, 1)

    if N > 0
        νN = ν0 + N
        Sμ = sum((X[i,:]-μ) * (X[i,:] - μ)' for i in 1:N)
        SN = S0 + Sμ
        return InverseWishart(νN, SN)
    else
        return InverseWishart(ν0, S0)
    end
end

function mu_var_gibbs(priors, X, k, draw)
    @unpack m0, V0, ν0, S0 = priors

    m = [draw.μ1, draw.μ2]
    s = [draw.Σ1, draw.Σ2]

    sigma_post = posterior_variance(ν0[k], S0[k], m[k], X)
    mu_post = posterior_mu(m0[k], V0[k], s[k], X)

    Σ = Symmetric(rand(sigma_post))
    μ = rand(mu_post)

    Σ_like = logpdf(sigma_post, Σ)
    μ_like = logpdf(mu_post, μ)

    # @info "Mu/Var" k μ Σ

    return (μ, Σ), (μ_like, Σ_like)
end

function emit_density(f, dist)
    return logpdf(dist, f)
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
    α[:,1] = log.(pi) .+ map(i -> emit_density(f[1,:], dists[i]), 1:K)
    # α[:,1] = α[:,1] ./ sum(α[:,1])
    α[:,1] = α[:,1] .- logsumexp(α[:,1])

    # Preallocate smoothing
    β = zeros(K, T)
    β[:,end] .= 0.0

    for k in 1:K
        predicted[1,1] += log(A[k,1]) + α[k, 1]
        predicted[2,1] += log(A[k,2]) + α[k, 1]
    end

    for t in 2:T
        α[1,t] = emit_density(f[t,:], dists[1])
        α[2,t] = emit_density(f[t,:], dists[2])

        for k in 1:K
            α[1,t] += log(A[k,1]) + α[k, t-1]
            α[2,t] += log(A[k,2]) + α[k, t-1]
        end

        α[:,t] = α[:,t] .- logsumexp(α[:,t])
        # α[:,t] = α[:,t] ./ sum(α[:,t])

        for k in 1:K
            predicted[1,t] += log(A[k,1]) + α[k, t]
            predicted[2,t] += log(A[k,2]) + α[k, t]
        end
    end

    for t in T-1:-1:1
        for k in 1:K
            β[k,t] += exp(α[k,t+1] + predicted[k, t] + β[k, t+1])
        end

        # β[:,t] = β[:,t] ./ sum(β[:,t])
        β[:,t] = β[:,t] .- logsumexp(β[:,t])
    end

    # display(maximum(α, dims=1))
    # α = exp.(α .- maximum(α, dims=1))
    # α = exp.(α .- maximum(α, dims=1))

    # display(β)
    ab = α .+ β
    smoothed = ab .- logsumexp(ab, dims=1)
    # display(plot(smoothed[1,:]))
    # display(smoothed)
    # display(α)
    # display(β)
    # display(smoothed)
    # error()

    return α, predicted, β, exp.(smoothed)
end

function posterior_states(A, pi, μs, Σs, X)
    α, predicted, β, smoothed = filtering(X, A, μs, Σs, pi)

    # println("========================")
    # display(α)
    # display(β)
    # foreach(display, μs)
    # foreach(display, Σs)
    # display(smoothed)
    dists = vec(mapslices(z -> Categorical(z), smoothed, dims=[1]))
    S = map(x -> rand(x), dists)
    state_like = sum(map(i -> logpdf(dists[i], S[i]), 1:length(S)))
    return S, state_like
end

function predicted_states(s::Int, A)
    K = size(A, 1)

    P = zeros(K)
    for k in 1:K
        P[k] = A[s, k]
    end

    return P
end

# Simulate HMM states
function hmm_states(A, T, pi)
    states = zeros(Int, T)
    states[1] = rand(Categorical(pi))

    ℓ = 0.0
    for t in 2:T
        d = Categorical(A[states[t-1], :])
        states[t] = rand(d)
        ℓ += logpdf(d, states[t])
    end

    return states, ℓ
end

# Make HMM emissions
function hmm_emit(states, mus, sigmas)
    f = zeros(length(states), length(mus[1]))
    dists = [MvNormal(mus[i], sigmas[i]) for i in 1:length(mus)]

    for i in 1:length(states)
        f[i, :] = rand(dists[states[i]])
    end

    return f
end

function conditional_payoffs(mus, sigmas, N, probs)
    dists = [MvNormal(mus[i], sigmas[i]) for i in 1:length(mus)]
    d = Categorical(probs)
    s = rand(d, N)
    draws = zeros(length(mus[1]), N)
    densities = map(z -> logpdf(d, z), s)

    for i in 1:N
        draws[:,i] = rand(dists[s[i]])
        densities[i] += logpdf(dists[s[i]], draws[:,i])
    end

    return draws, densities
end

function prices(delta, w, gamma, mus, sigmas, N, probs)
    draws, densities = conditional_payoffs(mus, sigmas, N, probs)
    m = w'exp.(draws) .^ (-gamma)
    P = vec(delta .* sum(densities .* (m .* draws), dims=2))
    R = sum(densities .* (draws ./ P), dims=2)

    @info "" P R
    return P, R
end

function initialize(priors; n_simulations=1_000)
    @unpack m0, V0, ν0, S0, delta, gamma, w, pi, A, T = priors
    K = length(m0)

    μ_priors = [MvNormal(m0[i], V0[i]) for i in 1:K]
    Σ_priors = [InverseWishart(ν0[i], S0[i]) for i in 1:K]

    μ = [rand(μ_priors[i]) for i in 1:K]
    Σ = [Symmetric(rand(Σ_priors[i])) for i in 1:K]
    S, ℓ = hmm_states(A, T, pi)

    for k in 1:K
        ℓ += logpdf(μ_priors[k], μ[k])
        ℓ += logpdf(Σ_priors[k], Σ[k])
    end

    # Calculate future prices
    P_state = predicted_states(S[end], A)
    price, returns = prices(delta, w, gamma, μ, Σ, n_simulations, P_state)

    return (
        μ1=μ[1], μ2=μ[2], 
        Σ1=Σ[1], Σ2=Σ[2], 
        S=S, 
        price=price, 
        returns=returns,
        lp=ℓ
    )
end

function gibbs_step(
    priors, 
    data, 
    draw; 
    override=NamedTuple(), 
    n_simulations=1_000
)
    @unpack delta, w, gamma, pi, A, T = priors
    K = size(A, 1)

    S, state_like = posterior_states(
        A, 
        pi, 
        [draw.μ1, draw.μ2], 
        [draw.Σ1, draw.Σ2], 
        data
    )
    res = map(k -> mu_var_gibbs(priors, data[S .== k, :], k, draw), 1:K)
    μ = map(t -> t[1][1], res)
    Σ = map(t -> t[1][2], res)
    μ_like = map(t -> t[2][1], res)
    Σ_like = map(t -> t[2][2], res)
    lp = sum(μ_like) + sum(Σ_like) + state_like

    # Calculate future prices
    P_state = predicted_states(S[end], A)
    price, returns = prices(delta, w, gamma, μ, Σ, n_simulations, P_state)

    return (
        μ1=μ[1], μ2=μ[2], 
        Σ1=Σ[1], Σ2=Σ[2], 
        S=S, 
        price=price, 
        returns=returns,
        lp=lp
    )
end

function add_riskfree(mu, sigma)
    mu_new = vcat(1.0, mu)
    M = size(sigma, 1)
    sigma_new = zeros(M+1, M+1)
    sigma_new[2:end, 2:end] = sigma
    sigma_new[1,1] = 0.00000001

    return mu_new, sigma_new
end

function expectation(values, lp)
    weights = rescale_lp(lp)

    return sum(values .* weights)
end

pi = [0.5, 0.5]
# pi = [0.5, 0.5]
A = [
    0.6 0.4;
    0.7 0.3
]
n_states = size(A, 1)

Random.seed!(2)
S_true, _ = hmm_states(A, 5, pi)
# S = [1,1,1,2,2]
# state_post = posterior_states(A, pi, mu, sigma, f)

# Stochastic parameters
n_assets = 1
iw = InverseWishart(n_assets + 2, diagm(ones(n_assets)))
mu = [randn(n_assets) .+ 1.015 for _ in 1:n_states]
sigma = [rand(iw) for _ in 1:n_states]

# Deterministic parameters
# Random.seed!(5)
# mu =[
#     [2, -0.5],
#     [-0.5, 0.25],
# ]
# sigma = [
#     [1.0 0.25; 0.25 1.0],
#     [2.0 -0.25; -0.25 2.0],
# ]
# n_assets = length(mu[1])

for i in 1:n_states
    mu[i], sigma[i] = add_riskfree(mu[i], sigma[i])
end
n_assets += 1

data = hmm_emit(S_true, mu, sigma)


m0 = [
    mu[1],
    mu[2]
]
# m0 = map(I -> zeros(n_assets), 1:n_states)

V0 = [
    0.0003I, 3I
]
# V0 = [
#     Float64[1 0; 0 1],
#     Float64[1 0; 0 1]
# ]

ν0 = [n_states+n_assets+1 for _ in 1:n_states]
# ν0 = [15, 15]
# S0 = [
#     sigma[1],
#     [1.0 -0.15; -0.15 1.0]
# ]
S0 = [
    diagm(ones(n_assets)) for j in 1:n_states
]
S0[1,1] /= 1000

priors = (
    A=A, 
    pi=pi, 
    μs=mu, 
    Σs=sigma, 
    m0=m0,
    V0=[1I, 1I], 
    ν0=ν0, 
    S0=S0,
    T=length(S_true),
    delta=0.95,
    w=[1/n_assets for i in 1:n_assets],
    gamma=2.0
)

draws = [initialize(priors)]

# for i in 2:200
@showprogress for i in 2:2000
    push!(draws, gibbs_step(priors, data, draws[i-1]))
end



S = mapreduce(x -> x.S', vcat, draws)
μ1 = map(x -> x.μ1, draws)
μ2 = map(x -> x.μ2, draws)
Σ1 = map(x -> x.Σ1, draws)
Σ2 = map(x -> x.Σ2, draws)
lp = map(x -> x.lp, draws)
P = mapreduce(x -> x.price', vcat, draws)
R = mapreduce(x -> x.returns', vcat, draws)
density(lp)

lp_scaled = exp.(lp .- maximum(lp))
lp_scaled = lp_scaled ./ sum(lp_scaled)

eq_prices = map(i -> expectation(P[:,i], lp), 1:n_assets)
eq_returns = map(i -> expectation(R[:,i], lp), 1:n_assets)

# draws = draws[100:end]

# μ1 = map(x -> x.μ[1], draws)
# μ11 = map(x -> x[1], μ1)
# μ12 = map(x -> x[2], μ1)

# μ2 = map(x -> x.μ[2], draws)
# μ21 = map(x -> x[1], μ2)
# μ22 = map(x -> x[2], μ2)

# estimated_mu = [
#     mean(μ11) mean(μ12);
#     mean(μ21) mean(μ22)
# ]

# println("Estimated")
# display(estimated_mu)

# println("True")
# display(hcat(mu...))

# p = plot(μ11)
# plot!(p, μ12)
# plot!(p, μ21)
# plot!(p, μ22)
# display(p)

# function state_summary(S)
#     max_states = maximum(S)
#     T = size(S, 1)
#     N = size(S, 2)
#     summ = zeros(T, max_states)
#     for i in 1:T
#         s = S[i,:]
#         for k in 1:max_states
#             summ[i,k] = sum(s .== k) / N
#         end
#     end

#     return summ
# end





# S_post = mapreduce(x -> x.S, hcat, draws)

# [S state_summary(S_post)]
# S_post = mean(map(x -> x.S, draws))

# sig_1 = mean(map(x -> x.Σ[1], draws))
# sig_2 = mean(map(x -> x.Σ[2], draws))

# summ = state_summary(S_post)
# sts = [S summ[:,1]]
# plot(sts) |> display

# df = DataFrame(
#     true_states = S,
#     state_1_post = summ[:,1],
#     state_2_post = summ[:,2],
# )

# P = mapreduce(x -> x.price', vcat, draws)

# p2 = plot(sts[:,1], label="Ground truth")
# plot!(p2, sts[:,2], label="Posterior mean")

# plot(p, p2)

# α, predicted, β, smoothed = filtering(f, A, mu, sigma, pi)

# display(α)
# display(predicted)
# display(β)
# display(smoothed)
# display(S')

# Test posterior mean code
# S_true =  Symmetric([2.0 0.0; 0.0 1.0])
# S_prior = [10.0 0.0; 0.0 10.0]
# mu_true = [1.5, -1.5]
# mu_prior = [0.0, 0.0]

# n = 1000
# data = rand(MvNormal(mu_true, S_true), n)'

# mu_post_dist = posterior_mu(mean(data, dims=1) |> vec, S_prior, S_true, data)

# # Test posterior variance code
# ν0 = 2
# var_post_dist = posterior_variance(ν0, cov(data), mu_true, data)

# # Report variances
# @info "Posteriors" mean(mu_post_dist) mean(var_post_dist)

# function sampling_func(draw=nothing)
#     return mu_var_gibbs(mu_prior, S_prior, ν0, S_prior, data; draw=draw)
# end

# transitions = [sampling_func()]

# for i in 2:10000
#     push!(transitions, sampling_func(transitions[i-1]))
# end

# μ1 = map(t -> t.μ[1], transitions)
# μ2 = map(t -> t.μ[2], transitions)

# plot(μ1)
# plot!(μ2)q