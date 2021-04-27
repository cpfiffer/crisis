# Imports
using Distributions
using UnicodePlots

# Set up consumption process
μ = [2, 1] # Mean consumption
σ = [1, 2] # Var consumption
A = [0.99 0.01; 0.05 0.95] # Transition matrix
B = [Normal(μ[1], σ[1]), Normal(μ[2], σ[1])] # Emission densities
π = [0.5, 0.5] # Initial state probabilities

# Emission
function emit(s)
    return rand(B[s])
end

transition() = rand(Categorical(π))
transition(s_prev) = rand(Categorical(A[s_prev, :]))

# Draw a consumption path
function consumption(T)
    c = zeros(T)
    s = zeros(Int, T)

    s[1] = transition()
    c[1] = emit(s[1])

    for t in 2:T
        s[t] = transition(s[t-1])
        c[t] = emit(s[t])
    end

    return c, s
end

c, s = consumption(100)

# Compute forward likelihood
function likelihood(c)
    # Initialize likelihood
    ℓ = 0

    # Get params
    T = length(c)
    N = length(B)

    # Create α matrix
    α = zeros(N, T)

    # Calculate the alpha for the first state
    α[:, 1] = π .* map(z -> pdf(z, c[1]), B)

    # Get all the later alphas
    for t in 2:T
        for i in 1:N
            for j in 1:N
                α[i, t] += α[j, t-1] * A[i,j]
            end

            α[i, t] *= pdf(B[i], c[t])
        end
    end

    return α
end

# likelihood(c)

function backward(c)
    T = length(c)
    N = length(B)

    β = zeros(N, T)

    β[:, end] = ones(N)

    for t in T-1:-1:1
        for i in 1:N
            for j in 1:N
                β[i, t] += A[i,j] * pdf(B[j], c[t+1]) * β[j, t+1]
            end
        end
    end

    α = likelihood(c)
    P = sum(α, dims=1)
    γ = α .* β ./ P

    return γ[:,end]
end

# prob = backward(c)

probs = map(i -> backward(c[1:i]), 1:length(c))

p1 = [x[1] for x in probs]
p2 = [x[2] for x in probs]

println("True state")
lineplot(s) |> display

println("State 1 probability")
lineplot(p1) |> display

println("State 2 probability")
lineplot(p2) |> display

#lineplot(c)
