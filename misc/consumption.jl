# Imports
using Distributions
using UnicodePlots
using Random
Random.seed!(1)

# Set up consumption process
μ = [2, 2] # Mean consumption
σ = [1, 0.1] # Var consumption
A = [0.6 0.4; 0.4 0.6] # Transition matrix
B = [Normal(μ[1], σ[1]), Normal(μ[2], σ[2])] # Emission densities
π = [0.5, 0.5] # Initial state probabilities

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

function MarkovianFiltration(
    N::Int, 
    T::Int, 
    π::Vector, 
    A::AbstractMatrix, 
    B::Vector{<:Function}
)
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
                mf.b[i, t] = mf.B[i](Y[t])
            end
        end
    end

    return mf.b[:, 1:M]
end

function forward!(mf::MarkovianFiltration, Y::Vector)
    M = length(Y)

    mf.α[:,1] = mf.π .* mf.b[:,1]

    for t in 1:M
        for i in 1:mf.N
            if ismissing(mf.α[i,t])
                mf.α[i,t] = 0
                for j in 1:mf.N
                    mf.α[i,t] += mf.α[j,t-1] * mf.A[i, j]
                end
                mf.α[i,t] *= mf.b[i,t]
            end
        end
        mf.α[:,t] /= sum(mf.α[:,t])
    end

    return mf.α
end

function backward!(mf::MarkovianFiltration, Y::Vector)
    M = length(Y)

    mf.β[:,M] .= 1

    for t in (M-1):-1:1
        for i in 1:mf.N
            for j in 1:mf.N
                mf.β[i,t] = mf.A[i,j] * mf.b[j, t+1] * mf.β[j, t+1]
            end
        end

        mf.β[:,t] /=  sum(mf.β[:,t])
    end

    return mf.β
end

function filtration(mf::MarkovianFiltration, Y::Vector)
   likelihood!(mf, Y) 
   forward!(mf, Y) 
   backward!(mf, Y)

   γ = mf.α .* mf.β ./ sum(mf.α, dims=1)

   return γ ./ sum(γ, dims=1)
end

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
function likelihood(c; rescale=false)
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

        if rescale
            α[:, t] /= sum(α[:, t], dims=1)
        end
    end

    # asum = sum(α, dims=1)

    return α
end

α = likelihood(c, rescale = true)

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

prob = backward(c)

# probs = map(i -> backward(c[1:i]), 1:length(c))

# p1 = [x[1] for x in probs]
# p2 = [x[2] for x in probs]

# println("Consumption")
# lineplot(c) |> display

#println("True state")
#lineplot(s) |> display

# println("State 1 probability")
# lineplot(p1) |> display

# println("State 2 probability")
# lineplot(p2) |> display

#lineplot(c)

mf = MarkovianFiltration(
    length(B),
    length(c),
    π,
    A,
    map(d -> (x -> pdf(d, x)), B)
)

m = 10
# likelihood!(mf, c[1:m])
# forward!(mf, c[1:m])
# backward!(mf, c[1:m])

display(prob)
display(filtration(mf, c))