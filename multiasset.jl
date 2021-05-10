# Imports
import Pkg; Pkg.activate(".")

# Setup
using Distributions
using Combinatorics

# Define a markov problem
function MarkovProblem(
    init_states,
    transition_matrix,
    consumption_densities,
    alphas,
    betas,
    firm_covs,
)
    return (
        init_states = init_states,
        transition_matrix = transition_matrix,
        consumption_densities = consumption_densities,
        alphas = alphas,
        betas = betas,
        firm_covs = firm_covs
    )
end

initial_state(problem) = convert(Int, rand(Categorical(problem.init_states)))
next_state(s::Int, problem) = convert(Int, rand(Categorical(problem.transition_matrix[s,:])))
consumption(s::Int, problem) = rand(problem.consumption_densities[s])

function firm_returns(Δc, s::Int, problem; noise=true)
    R = problem.alphas + problem.betas .* Δc
    if noise
        noise_draw = rand(MvNormal(problem.firm_covs[s]))
        return R + noise_draw
    else
        return R
    end
end

function transition(problem; noise=true)
    # Draw the first state
    s′ = initial_state(problem)

    # Calculate consumption
    Δc = consumption(s′, problem)

    # Calculate firm returns
    R = firm_returns(Δc, s′, problem; noise=noise)
    return s′, Δc, R
end

function transition(s::Int, problem; noise=true)
    # Draw a new state
    s′ = next_state(s, problem)

    # Calculate consumption
    Δc = consumption(s′, problem)

    # Calculate firm returns
    R = firm_returns(Δc, s′, problem; noise=noise)
    return s′, Δc, R
end

function simulate(problem, T)
    S = zeros(Int, T) # State
    C = zeros(T) # Consumption
    R = zeros(T, length(problem.alphas)) # Returns

    # First draw
    S[1], C[1], R[1,:] = transition(problem)

    # Subsequent draws
    for t in 2:T
        S[t], C[t], R[t,:] = transition(S[t-1], problem)
    end

    return (S=S, C=C, R=R)
end

prob_state(s, problem) = log(problem.init_states[s])
prob_state(s, s′, problem) = log(problem.transition_matrix[s,s′])

prob_consumption(s::Int, c::Float64, problem) = logpdf(problem.consumption_densities[s], c)

function prob_ret(s::Int, c::Float64, R::Vector, problem)
    ER = firm_returns(c, s, problem, noise=false)
    VR = problem.firm_covs[s]
    return logpdf(MvNormal(ER, VR), R)
end

function next_perm(m, nums)
    for i in length(m):-1:1
        if m[i] < nums
            m[i] += 1
            break
        else 
            m[i] = 1
        end         
    end

    return m
end

function state_perms(nums, l)
    # Preallocate
    total = nums ^ l
    M = ones(Int, total, l)

    for i in 2:total
        M[i, :] = next_perm(M[i-1,:], nums)
        # next_perm!(M[i-1,:], M[i, :], nums)
    end

    return M
end

function joint(S, C, R, problem)
    T = length(S)

    # Calculate state probs first
    sp_probs = [prob_state(S[1], problem); map(i -> prob_state(S[i-1], S[i], problem), 2:T)]
    sp = sum(sp_probs)

    sp_norm = sp_probs ./ sp
    @info "" sp_norm

    # Then calculate the conditional consumption/firm probs
    cp = 0.0
    fp = 0.0
    for t in 1:T
        s = S[t]
        c = C[t]
        r = R[t,:]
        cp += prob_consumption(s, c, problem)
        fp += prob_ret(s, c, r, problem)
    end

    p = exp(sp + cp + fp) 
    # p = p ./ sum(p)
    return p
end

# Create the problem
A = [0.9 0.1; 0.1 0.9]
Π = [0.5, 0.5]
μ = [1, 2]
σ = [1, 1]
ΔC = [Normal(μ[i], σ[i]) for i in 1:length(μ)]

# Firm properties
alphas = [0.05, 0.04]
betas = [1.5, 0.5]
good_cov = [1.0 0.1; 0.1 1.0]
bad_cov = [2.0 0.4; 0.4 2.0]

problem = MarkovProblem(Π, A, ΔC, alphas, betas, [bad_cov, good_cov])
# transition(1, problem)
# transition(1, problem; noise=false)

T = 5
S, C, R = simulate(problem, T)

joint(S, C, R, problem)

s_possible = state_perms(2, T)

map(s -> joint(s_possible[s, :], C, R, problem), 1:size(s_possible, 1))

# Function to transition
# function transition(s, transition_matrix, )

# end