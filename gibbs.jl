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
# using Optim
using JuMP

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
    # s = [draw.Σ1, draw.Σ2]

    sigma_post = posterior_variance(ν0[k], S0[k], m[k], X)
    Σ = Symmetric(rand(sigma_post))

    mu_post = posterior_mu(m0[k], V0[k], Σ, X)
    μ = rand(mu_post)

    Σ_like = logpdf(sigma_post, Σ)
    μ_like = logpdf(mu_post, μ)

    # @info "" μ mu_post k  m

    # @info "Mu/Var" k μ Σ

    return (μ, Σ), (μ_like, Σ_like)
end

function emit_density(f, dist)
    # return loglikelihood(dist, f)
    return logpdf(dist, f)
end

function filtering_nonlog(f, A, mus, sigmas, pi)
    K = size(A, 1)
    T = size(f, 1)
    dists = [MvNormal(mus[i], sigmas[i]) for i in 1:K]

    α = zeros(K,T)
    predicted = zeros(K, T)
    β = zeros(K, T)
    smoothed = zeros(K,T)
    
    α[1,1] = pi[1] * loglikelihood(dists[1], f[1,:])
    α[2,1] = pi[2] * loglikelihood(dists[2], f[1,:])
    α[:,1] = α[:,1] .- logsumexp(α[:,1])
    
    for t in 2:T
        corrector = [
            loglikelihood(dists[1], f[t,:]),
            loglikelihood(dists[2], f[t,:])
        ]

        predictor = [
            logsumexp(log(A[1,1]) + α[1,t-1], log(A[2,1]) + α[2,t-1]),
            logsumexp(log(A[1,2]) + α[1,t-1], log(A[2,2]) + α[2,t-1]),
        ]

        α[:,t] = corrector + predictor
        α[:,t] = α[:,t] .- logsumexp(α[:,t])
    end

    for t in T-1:-1:1
        β[1, t] = logsumexp(
            loglikelihood(dists[1], f[t+1,:]) + log(A[1,1]) + β[1,t+1],
            loglikelihood(dists[2], f[t+1,:]) + log(A[1,2]) + β[2,t+1],
        )

        β[2, t] = logsumexp(
            loglikelihood(dists[1], f[t+1,:]) + log(A[2,1]) + β[1,t+1],
            loglikelihood(dists[2], f[t+1,:]) + log(A[2,2]) + β[2,t+1],
        )

        β[:,t] = β[:,t] .- logsumexp(β[:,t])

    end

    ab = α + β
    smoothed = ab .- map(i -> logsumexp(ab[:,i]), 1:T)'

    return exp.(α), exp.(predicted), exp.(β), exp.(smoothed)
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

    init_emit = map(i -> emit_density(f[1,:], dists[i]), 1:K)
    α[:,1] = log.(pi) + init_emit
    α[:,1] = α[:,1] .- logsumexp(α[:,1])

    # α[:,1] = α[:,1] .- logsumexp(α[:,1])
    
    # Using non-logs
    # α[:,1] = pi .* map(i -> emit_density(f[1,:], dists[i]), 1:K)
    # display(α)
    # α[:,1] = α[:,1] ./ sum(α[:,1])

    # Preallocate smoothing
    β = zeros(K, T)
    β[:,end] .= 0.0

    for k in 1:K
        predicted[1,1] += log(A[k,1]) + α[k, 1]
        predicted[2,1] += log(A[k,2]) + α[k, 1]
    end
    predicted[:,1] = predicted[:,1] .- logsumexp(predicted[:,1])

    for t in 2:T
        α[1,t] = emit_density(f[t,:], dists[1])
        α[2,t] = emit_density(f[t,:], dists[2])

        # Log
        for k in 1:K
            α[1,t] += logsumexp(log(A[k,1]) + α[k, t-1])
            α[2,t] += logsumexp(log(A[k,2]) + α[k, t-1])
        end

        # Non-log
        # for k in 1:K
        #     α[1,t] *= A[k,1] * α[k, t-1]
        #     α[2,t] *= A[k,2] * α[k, t-1]
        # end

        # α[:,t] = α[:,t] .- logsumexp(α[:,t])
        # α[:,t] = α[:,t] ./ sum(α[:,t])
        α[:,t] = α[:,t] .- logsumexp(α[:,t])

        for k in 1:K
            predicted[1,t] += log(A[k,1]) + α[k, t]
            predicted[2,t] += log(A[k,2]) + α[k, t]
        end

        predicted[:,t] = predicted[:,t] .- logsumexp(predicted[:,t])
    end

    for t in T-1:-1:1
        for k in 1:K
            # β[k,t] += exp(α[k,t+1] + predicted[k, t]) + β[k, t+1]
            a = emit_density(f[t+1,:], dists[k])
            b = predicted[k, t+1]
            c = β[k, t+1]
            β[k,t] += logsumexp(a + b + c)
        end

        # β[:,t] = β[:,t] ./ sum(β[:,t])
        # β[:,t] = β[:,t] .- logsumexp(β[:,t])
    end

    ab = α .+ β
    smoothed = ab .- logsumexp(ab, dims=1)

    return exp.(α), exp.(predicted), exp.(β), exp.(smoothed)
end

function posterior_states(A, pi, μs, Σs, X)
    α, predicted, β, smoothed = filtering_nonlog(X, A, μs, Σs, pi)

    dists = vec(mapslices(z -> Categorical(z), smoothed, dims=[1]))
    S = map(x -> rand(x), dists)
    state_like = sum(map(i -> logpdf(dists[i], S[i]), 1:length(S)))
    return (
        S, 
        state_like, 
        (
            predicted=predicted, 
            smoothed=smoothed, 
            α=α,
            β=β
        )
    )
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

    return draws, rescale_lp(densities)
end

function port_payoff(w, draws, densities)
    payoffs = (w'draws)
    expectation = payoffs*densities

    variance = sum(w .* (payoffs .- expectation).^2)
    # expectation = (w'draws)
    # variance = ((w'draws) - expectation)^2 * densities
    return expectation, variance
end

function utility(w, gamma, draws, densities)
    m, s = port_payoff(w, draws, densities)

    return m - (gamma^2 / 2) * s
end

function optimal_weights(gamma, draws, densities)
    N = size(draws, 1)
    init_w = [1/N for i in 1:N]
    target(w...) = utility(collect(w), gamma, draws, densities)
    # target(w...) = sum(collect(w) .^ 2)

    model = Model(Ipopt.Optimizer)
    register(model, :target, N, target; autodiff = true)
    set_silent(model)
    @variable(model, w[1:N] >= -1)
    @NLobjective(model, Max, target(w...))
    @NLconstraint(model, sum(w[i] for i in 1:N) == 1)
    optimize!(model)
    
    return value.(w)
end

function prices(delta, w, gamma, mus, sigmas, N, probs)
    draws, densities = conditional_payoffs(mus, sigmas, N, probs)

    # res = optimal_weights(gamma, draws, densities)
    # display(res)
    # display(res.minimizer)
    # error()

    m = w'exp.(draws) .^ (-gamma)

    P = vec(delta .* (densities' * (m .* draws)'))

    # Rs = (draws ./ P)
    optimal_w = w
    # optimal_w = optimal_weights(gamma, Rs, densities)
    
    R = vec((densities' * (draws ./ P)')) .- 1

    return P, R, optimal_w
end

function initialize(priors; n_simulations=10_000)
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
    price, returns, optimal_w = prices(delta, w, gamma, μ, Σ, n_simulations, P_state)

    return (
        μ1=μ[1], μ2=μ[2], 
        Σ1=Σ[1], Σ2=Σ[2], 
        S=S, 
        price=price, 
        returns=returns,
        w=optimal_w,
        lp=ℓ
    )
end

function gibbs_step(
    priors, 
    data, 
    draw; 
    override=NamedTuple(), 
    n_simulations=10_000
)
    @unpack delta, w, gamma, pi, A = priors
    K = size(A, 1)

    S, state_like, state_info = posterior_states(
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
    price, returns, optimal_w = prices(delta, w, gamma, μ, Σ, n_simulations, P_state)

    return (
        μ1=μ[1], μ2=μ[2], 
        Σ1=Σ[1], Σ2=Σ[2], 
        S=S, 
        price=price, 
        returns=returns,
        w=optimal_w,
        lp=lp,
        state_info = state_info
    )
end

function add_riskfree(mu, sigma)
    mu_new = vcat(1.0, mu)
    M = size(sigma, 1)
    sigma_new = zeros(M+1, M+1)
    sigma_new[2:end, 2:end] = sigma
    sigma_new[1,1] = 0.000000001

    return mu_new, sigma_new
end

function state_summary(S, max_states)
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

function expectation(values, lp)
    weights = rescale_lp(lp)

    return sum(values .* weights)
end

pi = [0.99, 0.01]
# pi = [0.5, 0.5]
A = [
    0.99 0.01;
    0.40 0.60
]
n_states = size(A, 1)

Random.seed!(2)
# S_true, _ = hmm_states(A, 500, pi)
S_true = vcat(repeat([1], 10), repeat([2], 10))
# state_post = posterior_states(A, pi, mu, sigma, f)

# Stochastic parameters
n_assets = 1
iw = InverseWishart(n_assets + 10, diagm(ones(n_assets)))
mu = [randn(n_assets) for _ in 1:n_states]
mu[1] = mu[1] .+ 6
mu[2] = mu[2] .+ 3
sigma = [rand(iw) for _ in 1:n_states]

# Deterministic parameters
# Random.seed!(5)
# mu =[
#     [1.5, 1.5],
#     [1.25, 1.25],
# ]
# sigma = [
#     [1.0 0.25; 0.25 1.0],
#     [2.0 -0.25; -0.25 1.0],
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
    0.05 .* diagm(ones(n_assets)), 
    0.05 .* diagm(ones(n_assets))
]
# V0[1][1,1] = 0.000001
# V0[2][1,1] = 0.000001
# V0 = [
#     Float64[1 0; 0 1],
#     Float64[1 0; 0 1]
# ]

ν0 = [n_states+n_assets+1 for _ in 1:n_states]
# ν0 = [15, 15]
# S0 = [
#     sigma[1],
#     sigma[2]
# ]
S0 = [
    diagm(ones(n_assets)) for j in 1:n_states
]
S0[1][1,1] /= 100000
S0[2][1,1] /= 100000

dfs = []
state_dfs = []
# t = length(S_true)

for t in 1:length(S_true)
    priors = (
        A=A, 
        pi=pi, 
        μs=mu, 
        Σs=sigma, 
        m0=m0,
        V0=V0, 
        ν0=ν0, 
        S0=S0,
        T=t,
        delta=1.0,
        w=[1/n_assets for i in 1:n_assets],
        gamma=1
    )

    draws = []

    d1 = initialize(priors)
    push!(draws, d1)

    # for i in 2:200
    @showprogress for i in 2:10_000
        push!(draws, gibbs_step(priors, data[1:t,:], draws[i-1]))
    end
    draws = draws[2:end]

    S = mapreduce(x -> x.S', vcat, draws)
    μ1 = map(x -> x.μ1, draws)
    μ2 = map(x -> x.μ2, draws)
    Σ1 = map(x -> x.Σ1, draws)
    Σ2 = map(x -> x.Σ2, draws)
    lp = map(x -> x.lp, draws)
    P = mapreduce(x -> x.price', vcat, draws)
    R = mapreduce(x -> x.returns', vcat, draws)
    w = mapreduce(x -> x.w', vcat, draws)
    # smoothed = map(x -> x.state_info.smoothed, draws)
    # α = map(x -> x.state_info.α, draws)
    # β = map(x -> x.state_info.β, draws)
    # predicted = map(x -> x.state_info.predicted, draws)

    # muplot = plot(map(x -> x[1], μ1))
    # plot!(muplot, map(x -> x[2], μ1))
    # plot!(muplot, map(x -> x[1], μ2))
    # plot!(muplot, map(x -> x[2], μ2))
    # display(muplot)

    # plot(map(x -> x[1], μ1)) |> display
    # plot(S[1,:]) |> display

    # println("\nSmoothed")
    # display(smoothed)
    # println("\nPredicted")
    # display(predicted)
    # println("\nα")
    # display(α)
    # println("\nβ")
    # display(β)
    # density(lp)

    lp_scaled = exp.(lp .- maximum(lp))
    lp_scaled = lp_scaled ./ sum(lp_scaled)

    eq_prices = map(i -> expectation(P[:,i], lp), 1:n_assets)
    eq_returns = map(i -> expectation(R[:,i], lp), 1:n_assets)
    risk_premium = eq_returns .- eq_returns[1]

    pstates = state_summary(S', size(A, 1))

    df = DataFrame(
        price = eq_prices,
        returns = eq_returns,
        premia = risk_premium,
        payoff_1 = mu[1],
        payoff_2 = mu[2],
        p_state_1 = pstates[end,1],
        p_state_2 = pstates[end,2],
        asset_id = 1:n_assets,
        t = t
    )

    m = vec(priors.w'exp.(data[1:t, :]') .^ (-priors.gamma))
    sdf = DataFrame(
        consumption = vec(priors.w'data[1:t, :]'),
        sdf = m,
        stateprob_1 = pstates[:,1],
        stateprob_2 = pstates[:,2],
        t = 1:t,
        iter = t,
    )

    push!(dfs, df)
    push!(state_dfs, sdf)
end

results = vcat(dfs...)
sdf_results = vcat(state_dfs...)
# risk_free = results[results.asset_id .== 2, :]

p1 = @df results plot(:t, :price, group=:asset_id)

!isdir("plots") && mkpath("plots")
savefig(p1, "plots/prices.png")
