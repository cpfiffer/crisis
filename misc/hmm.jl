# Imports
using LinearAlgebra, Distributions, Combinatorics

# Define system
## Transition matrix
A = [0.7 0.3; 0.4 0.6]

## Observation matrix
B = [0.1 0.4 0.5; 0.7 0.2 0.1]

## Initial states
π = [0.0, 1.0]

# Function to draw states
function hmm(N)
    # Store draws
    X = Vector{Int}(undef, N)
    O = Vector{Int}(undef, N)

    # Draw the first
    X[1] = rand(Categorical(π))
    O[1] = rand(Categorical(B[X[1], :]))

    # Draw all subsequent states
    for t in 2:N
        a = A[X[t-1], :]
        X[t] = rand(Categorical(a))
        O[t] = rand(Categorical(B[X[t], :]))
    end

    return (state=X, observations=O)
end

hmm(2)

# Score an HMM
obs = [2, 1, 3]

function score(observations)
    T = length(observations)
    N, M = size(B)

    possibles = repeat(collect(1:N), T)
    states = unique(collect(permutations(possibles, T)))
    ℓ = zeros(length(states))

    for (i, state) in enumerate(states)
        ℓ[i] = π[state[1]] * B[state[1], observations[1]]
        for t in 2:T
            s_prev = state[t-1]
            s = state[t]
            o = observations[t]
            prob_s = A[s_prev, s] * B[s, o]
            ℓ[i] *= prob_s
        end
    end

    return ℓ, sum(ℓ)
end

score(obs)
