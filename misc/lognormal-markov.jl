using Distributions
using StatsPlots

# Simulate states
A = [0.95 0.05; 0.05 0.95]
# mu = [1.0, 5.0]
# σ = [0.5, 0.5]
# Π = [0.5, 0.5]
# dists = [Normal(mu[i], σ[i]) for i in 1:length(mu)]

# In matrix form
μ_good = [0.1, 0.1]
Σ_good = [0.01 0.0; 0.0 0.01]

# μ_bad = [-0.1, -0.3]
μ_bad = [0.1, 0.1]
Σ_bad = [0.05 0.0; 0.0 0.15]

μ = [μ_good, μ_bad]

β = [2.0]

dists = [
    MvNormal(Σ_good),
    MvNormal(Σ_bad)
]

function returns(s, N)
    ε = rand(dists[s])

    # Calculate market return
    R = zeros(N)
    R[1] = μ[s][1] + ε[1]

    # Calculate firm returns
    R[2:end] = μ[s][2:end] + β .* R[1] + ε[2:end]

    return R
end

function simulate(T, n_sims, n_firms)
    s = zeros(Int64, T)
    R = zeros(T, n_firms)

    s[1] = rand(Categorical(Π))

    for t in 2:T
        s[t] = rand(Categorical(A[s[t-1], :]))
    end

    for t in 1:T
        R[t, :] = returns(s[t], n_firms)
    end

    return s, R
end

s, R = simulate(100, length(μ_bad))

# μs = [μ[i][1] for i in s]
# plot(s)
# plot!(R)

# function simulate_consumption(T, N)
#     smat = zeros(N, T)

#     for n in 1:N
#         smat[n, :] = simulate(T)
#     end

#     return smat
# end

# C = simulate_consumption(10, 200000)

# p = density()
# for t in 1:size(C, 2)
#     density!(p, C[:,t])
# end

# display(p)
