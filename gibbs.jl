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

    Σ = Symmetric(rand(posterior_variance(ν0[k], S0[k], draw.μ[k], X)))
    μ = rand(posterior_mu(m0[k], V0[k], draw.Σ[k], X))

    # @info "Mu/Var" k μ Σ

    return (μ, Σ)
end

function emit_density(f, dist)
    return loglikelihood(dist, f)
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
    return map(x -> rand(x), dists)
end

# Simulate HMM states
function hmm_states(A, T, pi)
    states = zeros(Int, T)
    states[1] = rand(Categorical(pi))
    for t in 2:T
        states[t] = rand(Categorical(A[states[t-1], :]))
    end

    return states
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

function initialize(m0, V0, ν0, S0, pi, A, T)
    μ = [rand(MvNormal(m0[i], V0[i])) for i in 1:length(m0)]
    Σ = [Symmetric(
            rand(InverseWishart(ν0[i], S0[i]))) for i in 1:length(ν0)]
    S = hmm_states(A, T, pi)

    return (μ=μ, Σ=Σ, S=S)
end

function gibbs_step(priors, data, draw; override=NamedTuple())
    @unpack m0, V0, ν0, S0, pi, A, T = priors
    K = size(A, 1)

    S = if :S in keys(override)
        override.S
    else
        posterior_states(A, pi, draw.μ, draw.Σ, data)
    end

    res = map(k -> mu_var_gibbs(priors, data[S .== k, :], k, draw), 1:K)
    μ = map(t -> t[1], res)
    Σ = map(t -> t[2], res)

    return (μ=μ, Σ=Σ, S=S)
end

pi = [0.5, 0.5]
# pi = [0.5, 0.5]
A = [
    0.95 0.05;
    0.7 0.3
]
n_states = size(A, 1)

Random.seed!(2)
S = hmm_states(A, 20, pi)
# S = [1,1,1,2,2]
# state_post = posterior_states(A, pi, mu, sigma, f)

# Stochastic parameters
n_assets = 50
iw = InverseWishart(n_assets + 2, diagm(ones(n_assets)))
mu = [randn(n_assets) .+ 1 for _ in 1:n_states]
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

data = hmm_emit(S, mu, sigma)


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

priors = (
    A=A, 
    pi=pi, 
    μs=mu, 
    Σs=sigma, 
    m0=m0,
    V0=[1I, 1I], 
    ν0=ν0, 
    S0=S0,
    T=length(S)
)

draws = [initialize(m0, V0, ν0, S0, pi, A, priors.T)]

or = override=(S=S,)
@showprogress for i in 2:2000
    push!(draws, gibbs_step(priors, data, draws[i-1]))
end

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

# p = plot(μ11)
# plot!(p, μ12)
# plot!(p, μ21)
# plot!(p, μ22)
# display(p)

function state_summary(S)
    max_states = maximum(S)
    T = size(S, 1)
    N = size(S, 2)
    summ = zeros(T, max_states)
    for i in 1:T
        s = S[i,:]
        for k in 1:max_states
            summ[i,k] = sum(s .== k) / N
        end
    end

    return summ
end

println("Estimated")
display(estimated_mu)

println("True")
display(hcat(mu...))



S_post = mapreduce(x -> x.S, hcat, draws)

# [S state_summary(S_post)]
# S_post = mean(map(x -> x.S, draws))

sig_1 = mean(map(x -> x.Σ[1], draws))
sig_2 = mean(map(x -> x.Σ[2], draws))

summ = state_summary(S_post)
sts = [S summ[:,1]]
plot(sts) |> display

df = DataFrame(
    true_states = S,
    state_1_post = summ[:,1],
    state_2_post = summ[:,2],
)

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
# plot!(μ2)