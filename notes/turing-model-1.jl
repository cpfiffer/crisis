# using Turing
using LinearAlgebra
using Distributions
using Random
using Plots

num_states = 2
num_assets = 10
num_periods = 10
num_sims = 1000

π = [0.5, 0.5]
A = [0.95 0.05; 0.05 0.95]

next_state(A, s) = rand(Categorical(A[s, :]))
state_prob(A, s1, s2) = log(A[s1,s2])

mutable struct MarkovianFiltration{V<:AbstractVector, M}
    N::Int            # Number of states
    T::Int            # Length of observation sequence
    π::V              # Initial state density
    α::AbstractMatrix # alpha pass matrix
    β::AbstractMatrix # beta pass matrix
    A::AbstractMatrix # Transition matrix
    b::AbstractMatrix # Cached likelihoods
    B::M              # Emission densities
end

function MarkovianFiltration(N::Int, T::Int, π::Vector, A::AbstractMatrix, B::Vector{<:Function})
    α = Array{Union{Missing, Float64}}(undef, N, T)
    β = Array{Union{Missing, Float64}}(undef, N, T)
    b = Array{Union{Missing, Float64}}(undef, N, T)

    α[:] .= missing
    β[:] .= missing
    b[:] .= missing

    return MarkovianFiltration(N, T, π, α, β, A, b, B)
end

"""
Computes the likelihood matrix using the function

b(k,j) = P(observation k at time t | state qj at time t)
"""
function likelihood!(mf::MarkovianFiltration, Y::Vector)
    M = length(Y)

    for i in 1:mf.N
        for t in 1:M
            if ismissing(mf.b[i,t])
                mf.b[i, t] = mf.B[i](Y[i])
            end
        end
    end

    return mf.b[:, 1:M]
end

function forward!(mf::MarkovianFiltration, Y::Vector)
    M = length(Y)

    for i in 1:mf.N
        mf.α[i,1] = mf.π .* mf.b[i,1]
        for t in 2:M
            if ismissing(mf.α[i,t])
                mf.α[i,t] = sum(mf.α[:,t-1] .* mf.A[:, i])
            end
        end
    end

    return mf.α[i,t]
end

function PS(A, S)
    ℓ = log(π[S[1]])
    ℓ += mapreduce(t -> state_prob(A, S[t-1], S[t]), +, 2:length(S))
    return ℓ
end

function PRS(S, r, means, sigmas; partial_obs = size(r, 2) .* ones(Int, size(r, 1)))
    ℓ = 0.0
    for t in 1:length(S)
        eind = partial_obs[t]
        s = S[t]
        m = means[s][1:eind]
        sigma = sigmas[s][1:eind, 1:eind]
        
        ℓ += logpdf(MvNormal(m, sigma), r[t,1:eind])
    end

    return ℓ
end

function Pjoint(S, R, means, sigmas)
    return PS(A, S) + PRS(S, R, means, sigmas)
end

function make_states(π, A, T)
    states = ones(Int, num_periods)
    states[1] = rand(Categorical(π))

    for t in 2:num_periods
        states[t] = next_state(A, states[t-1])
    end

    return states
end

function make_returns(states, means, sigmas, N)
    returns = zeros(length(states), N)
    returns[1, :] = rand(MvNormal(means[states[1]], sigmas[states[1]]))

    for t in 2:num_periods
        returns[t, :] = rand(MvNormal(means[states[t]], sigmas[states[t]]))
    end

    return returns
end

function MAP_state(
    A, R, means, sigmas; 
    partial_obs = size(R, 2) .* ones(Int, size(R, 1))
)
    T = size(R, 1)
    Q = size(A, 1)
    sts = Base.product((1:Q for _ in 1:T)...)
    
    max_ℓ = -Inf
    max_state = missing
    for (i, st) in enumerate(sts)
        enud = [j for j in st]
        ps = PS(A, enud)
        prs = PRS(enud, R, means, sigmas; partial_obs = partial_obs)

        ℓ = ps + prs
        if ℓ >= max_ℓ
            max_ℓ = ℓ
            max_state = enud
        end
    end

    return max_ℓ, max_state
end

function probs(A, R, means, sigmas; partial_obs = size(R, 2) .* ones(Int, size(R, 1)))
    T = size(R, 1)
    Q = size(A, 1)
    sts = Base.product((1:Q for _ in 1:T)...)
    stateslikes = zeros(length(sts))

    for (i, st) in enumerate(sts)
        enud = [j for j in st]
        ps = PS(A, enud)
        prs = PRS(enud, R, means, sigmas; partial_obs = partial_obs)
        stateslikes[i] = ps + prs
    end 

    elike = exp.(stateslikes)
    P = elike ./ sum(elike)

    return P
end

part_ob(T, N, n) = let 
    o = N .* ones(Int, T)
    o[end] = n
    return o
end

function KL(P, Q)
    return mapreduce(a -> a[1] * (log(a[1]) - log(a[2])), +, zip(P, Q))
end

global kls = zeros(num_assets-1, num_sims)
global topjoints = zeros(num_sims)

for draw in 1:num_sims
    sigma_dist = InverseWishart(num_assets + 30, diagm(ones(num_assets)))

    sigmas = [rand(sigma_dist) for _ in 1:num_states]
    means = [sigmas[i]*randn(num_assets) for i in 1:num_states]

    states = make_states(π, A, num_periods)
    returns = make_returns(states, means, sigmas, num_assets)

    # Just get the likelihood for the real state
    # topjoints[draw] = Pjoint(states, returns, means, sigmas)

    # Get the MAP state
    # mstate = MAP_state(A, returns, means, sigmas)
    # sse = sum((states - mstate[2]).^2)
    # display(sse)

    # Generate full changing densities
    densities = map(
        i -> probs(A, returns, means, sigmas, partial_obs = part_ob(num_periods, num_assets, i)), 
        1:num_assets
    )
    kls[:, draw] = map(i -> KL(densities[i], densities[i-1]), 2:num_assets)
end