# Imports
import Pkg; Pkg.activate(".")

# Setup
using Distributions
using Combinatorics
using LinearAlgebra
using Turing
using StatsPlots

# Define a markov problem
function MarkovProblem(
    init_states,
    transition_matrix,
    alphas,
    betas,
    covs,
    dists
)
    return (
        init_states = init_states,
        transition_matrix = transition_matrix,
        alphas = alphas,
        betas = betas,
        covs = covs,
        dists = dists
    )
end

initial_state(problem) = convert(Int, rand(Categorical(problem.init_states)))
next_state(s::Int, problem) = convert(Int, rand(Categorical(problem.transition_matrix[s,:])))

function firm_returns(s::Int, problem; noise=true)
    if noise
        return rand(problem.dists[s])
    else
        return problem.dists[s].μ
    end
end

function transition(problem; noise=true)
    # Draw the first state
    s′ = initial_state(problem)

    # Calculate firm returns
    R = firm_returns(s′, problem; noise=noise)

    return s′, R
end

function transition(s::Int, problem; noise=true)
    # Draw a new state
    s′ = next_state(s, problem)

    # Calculate firm returns
    R = firm_returns(s′, problem; noise=noise)

    return s′, R
end

function simulate(problem, T)
    S = zeros(Int, T) # State
    R = zeros(T, length(problem.alphas)) # Returns

    # First draw
    S[1], R[1,:] = transition(problem)

    # Subsequent draws
    for t in 2:T
        S[t], R[t,:] = transition(S[t-1], problem)
    end

    return (S=S, R=R)
end

# Create the problem
A = [0.9 0.1; 0.1 0.9] # Transition
Π = [0.9, 0.1] # Initial states

# Firm properties
alphas = [0.05, 0.04, 0.03]
betas = [1.5, 0.5]
good_cov = diagm(ones(length(alphas)))
bad_cov = diagm(3.0 .* ones(length(alphas)))
Γ = diagm(ones(length(alphas)))
Γ[1:(end-1), end] = betas
Γ_inv = inv(Γ)

good_dist = MvNormal(Γ_inv * alphas, good_cov)
bad_dist = MvNormal(Γ_inv * alphas, bad_cov)

covs = [good_cov, bad_cov]
problem = MarkovProblem(Π, A, alphas, betas, covs, [good_dist, bad_dist])
# transition(1, problem; noise=false)

T = 30
S, R = simulate(problem, T)

# Turing model
@model function hmm(R)
    T = size(R, 1)
    S = TArray{Int}(undef, T)

    S[1] ~ Categorical(Π)

    for t in 2:T
        S[t] ~ Categorical(A[S[t-1],:])
        R[t,:] ~ MvNormal(Γ_inv * alphas, covs[S[t]])
    end
end

# chain1 = sample(hmm(R[1:T-1, :]), SMC(), 10_000)
# chain2 = sample(hmm(R[1:T,:]), SMC(), 10_000)

p = plot(S .- 1)
for t in 2:T
    c = sample(hmm(R[1:t, :]), SMC(), 10_000)
    states = convert.(Int, group(c, "S").value.data) .- 1
    probs = mean(states, dims=1)[1,:,1]

    plot!(p, probs)
end

display(p)

# states1 = convert.(Int, group(chain1, "S").value.data) .- 1
# probs1 = mean(states1, dims=1)

# states2 = convert.(Int, group(chain2, "S").value.data) .- 1
# probs2 = mean(states2, dims=1)

# plot(probs1[1,:,1])
# plot!(probs2[1,:,1])
# plot!(S .- 1)
