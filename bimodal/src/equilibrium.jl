using Turing
using LinearAlgebra
using Optim
using StatsPlots
using Polynomials
using StatsFuns
using QuadGK
using Bijectors
using Random

Random.seed!(0)

# Construct Gamma matrix.
function make_gamma(b)
    b_new = [b; 1]
    N = length(b) + 1
    Γ = zeros(N,N)
    Γ[diagind(Γ)] .= 1
    Γ[:,end] = b_new

    return Γ
end

function make_mixture(μ, Σ)
    return MixtureModel([MvNormal(μ[i], Σ[i]) for i in 1:length(μ)])
end

# Calculate posterior risk factor variance & expectation
function risk_expectation(s_prob, μ)
    return s_prob .* μ[2] + (1-s_prob) .* μ[1]
end

function risk_variance(s_prob, Σ)
    return s_prob .* Σ[2] + (1-s_prob) .* Σ[1]
end

# State 2 probability
function conditional_posterior_state(ρ, μ, Σ, p, prior_s, η_j)

    ph = log(prior_s) + logpdf(MvNormal(μ[2], Σ[2]), p)
    pl = log(1-prior_s) + logpdf(MvNormal(μ[1], Σ[1]), p)
    denom = logsumexp(ph, pl)
    ph = exp(ph - denom)
    pl = 1 - ph

    @info "" ph pl p

    return [ph, pl]
    # return exp.([ph - denom, pl - denom])
end

# Quantity demanded function
function q(ρ, μ, Σ, p, r, η_j, prior_s)
    # posterior_s = posterior_s(ρ, μ, Σ, p, η_j)
    posterior_s = conditional_posterior_state(ρ, μ, Σ, p, prior_s, η_j)
    Vf = risk_variance(posterior_s[1], Σ)
    Ef = risk_expectation(posterior_s[1], μ)
    # Vf = risk_variance(s_prob, Σ)
    # Ef = risk_expectation(s_prob, μ)

    return 1/ρ .* inv(Vf) * (Ef - p .* r)
end

function eq_price(ρ, r, μ, Σ, s_prob, x, η_j)
    target(p) = sum((q(ρ, μ, Σ, p, r, η_j, s_prob) - x) .^ 2)

    return optimize(target, deepcopy(μ[1]), Newton())
end

@model function noisy_eq_nosignal(
    x_bar, 
    sigma_x,
    μ,
    Σ,
    ρ,
    s_prob,
    Γ,
    K;
    r = 1.0
)
    # Informational variables
    n_assets = length(μ[1])

    # Underlying state
    s ~ Bernoulli(s_prob) # s=1 means state 2, s=0 means state 1.

    # Risk factor shocks
    z ~ MvNormal(zeros(n_assets), Σ[s+1])
    f = inv(Γ) * μ[s+1] + z # Actual payoff

    # Risk factor supply
    x ~ MvNormal(x_bar, diagm(sigma_x))

    # Generate signals
    η_j ~ MvNormal(z, diagm(ones(n_assets) .* 1/K))

    # Calculate price
    p = eq_price(ρ, r, μ, Σ, s_prob, x)
end

# Parameters
ρ = 2
r = 1.0
n_states = 2
n_assets = 2
s_prob = 0.5
x_bar = [0.5, 10.0]
sigma_x = [1.0, 1.0]

# Random supply draw
rand_x = rand(MvNormal(x_bar, diagm(sigma_x)))

# z covariance structure.
iw = InverseWishart(n_assets + 3, diagm(ones(n_assets)))

Σ = [rand(iw) .* 3 for i in 1:n_states]
μ = [randn(n_assets) .* 3 .+ 5 for i in 1:n_states]

# Risk factor loadings
b = randn(n_assets - 1)
Γ = make_gamma(b)

risk_variance(s_prob, Σ)
risk_expectation(s_prob, μ)

# p = eq_price(ρ, r, μ, Σ, s_prob, rand_x, missing)

# Testing decomposition intuition
function eigenstuff(S)
    Σ_eigen = eigen(S)
    data = rand(MvNormal(zeros(n_assets), S), 10000)
    G = Σ_eigen.vectors
    L = diagm(Σ_eigen.values)

    inverted = data'inv(G)

    @info "" S G L G*L*G' cov(inverted)

    return L
end

function gmm_covar(μ, Σ)
    mm = make_mixture(μ, Σ)
    data = rand(mm, 100000)

    weights = mm.prior.p
    mu_bar = sum(weights .* μ)
    direct_cov = sum(weights[i] .* Σ[i] for i in 1:length(Σ))
    mean_diff_term = sum(weights[i] .* (μ[i] - mu_bar)*(μ[i] - mu_bar)' for i in 1:length(Σ))
    C = direct_cov + mean_diff_term

    return C
end

L1 = eigenstuff(Σ[1])
L2 = eigenstuff(Σ[2])
C = gmm_covar(μ, Σ)
L3 = eigenstuff(C)


# plot(z -> pdf(mm, z), -1:1, -1:1)

# Sampling
# model = noisy_eq_nosignal(    
#     x_bar, 
#     sigma_x,
#     μ,
#     Σ,
#     ρ,
#     s_prob,
#     1
# )
# sampler = Prior()

# chain = sample(model, sampler, 10_000)
# plot(chain)