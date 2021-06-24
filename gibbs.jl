using Distributions
using LinearAlgebra
using Statistics
using Parameters
using StatsPlots
using Random

function posterior_mu(m0, V0, Σ, X)
    μ = vec(mean(X; dims=1))
    N = size(X, 1)
    V0_inv = inv(V0)
    Σ_inv = inv(Σ)
    VN = Symmetric(inv(V0_inv + N .* Σ_inv))
    mN = VN * (Σ_inv * (N .* μ)  + V0_inv * m0)
    return MvNormal(mN, VN)
end

function posterior_variance(ν0, S0, μ, X)
    N = size(X, 1)
    νN = ν0 + N
    Sμ = sum((X[i,:]-μ) * (X[i,:] - μ)' for i in 1:N)
    SN = S0 + Sμ
    return InverseWishart(νN, SN)
end

function mu_var_gibbs(priors, X; draw=nothing)
    @unpack m0, V0, ν0, S0 = priors
    local μ,Σ

    if isnothing(draw)
        μ = rand(MvNormal(m0, V0))
        Σ = Symmetric(rand(InverseWishart(ν0, S0)))
    else
        @unpack μ, Σ = draw
    end

    Σ = Symmetric(rand(posterior_variance(ν0, S0, μ, X)))
    μ = rand(posterior_mu(m0, V0, Σ, X))

    return (μ, Σ)
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

    # Preallocate smoothing
    β = zeros(K, T)
    β[:,end] .= 1.0

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

    for t in T-1:-1:1
        for k in 1:K
            β[k,t] += exp(log(α[k,t+1]) + log(predicted[k, t]) + log(β[k, t+1]))
        end

        β[:,t] = β[:,t] ./ sum(β[:,t])
    end

    ab = α .* β
    smoothed = ab ./ sum(ab, dims=1)

    return α, predicted, β, smoothed
end

function posterior_states(A, pi, μs, Σs, X)
    α, predicted, β, smoothed = filtering(X, A, μs, Σs, pi)
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

function gibbs_step(priors, data)
    S = posterior_states(priors, data)
    μ_H, Σ_H = mu_var_gibbs(priors, data[S .== 1, :])
    μ_L, Σ_L = mu_var_gibbs(priors, data[S .== 2, :])
end

# Deterministic parameters
Random.seed!(5)
mu =[
    [1.5, 2.2],
    [2, 0.0],
] ./ 5
sigma = [
    [1.0 0.25; 0.25 1.0],
    [1.0 -0.22; -0.22 1.0],
]

pi = [0.95, 0.05]
A = [
    0.90 0.10;
    0.01 0.99
]

S = hmm_states(A, 10, pi)
f = hmm_emit(S, mu, sigma)
# state_post = posterior_states(A, pi, mu, sigma, f)

priors = (
    A=A, 
    pi=pi, 
    μs=mu, 
    Σs=sigma, 
    m0=m0, 
    V0=V0, 
    ν0=ν0, 
    S0=S0,
)
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