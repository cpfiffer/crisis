using Turing
using LinearAlgebra
using Optim
using StatsPlots
using Polynomials
using StatsFuns
using QuadGK
using Bijectors
using Random
using HCubature

Random.seed!(0)

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


function joint_density(f, ρ, r, μ, Σ, s_prob, x, s, η_j, σ_k, xbar, σ_x)
    d1 = MvNormal(f, diagm(σ_k))
    m1 = make_mixture(μ, Σ)
    x_dist = MvNormal(xbar, diagm(σ_x))

    ℓ = log(s_prob) + log(1-s_prob)
    
    ℓ += loglikelihood(m1, f)
    ℓ += loglikelihood(d1, η_j)
    ℓ += loglikelihood(x_dist, xbar + x)

    return ℓ
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

Σ = [rand(iw) .* 5 for i in 1:n_states]
μ = [randn(n_assets) .+ randn() for i in 1:n_states]

# Risk factor loadings
b = randn(n_assets - 1)

risk_variance(s_prob, Σ)
risk_expectation(s_prob, μ)

f = [4.0, 5.0]
x = [0.0, 0.0]
η_j = [5.6, 6.8]
σ_k = [2.0, 2.0]
xbar = [1.0, 1.0]



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

# Exact analytic covariance of GMM
# https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures
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

function wealth_transition(W0, r, q, f, p)
    return (f .- p' .* r) * q
end

function utility(rho, w)
    return exp(-rho * w)
end

function max_utility(rho, μ, Σ; draws = 10_000)
    mm = make_mixture(μ, Σ)
    data = rand(mm, draws)'
    d = vec(pdf(mm, data'))

    q = ones(n_assets) .* 1/n_assets
    p = mean(mm)

    b = bijector(Dirichlet(q))
    b_inv = inv(b)

    wj(q) = d'map(w -> utility(rho, w), vec(wealth_transition(0.0, 1.0, b_inv(q), data, p)))

    res = optimize(wj, q, LBFGS())

    display(res)

    display(eigen(cov(data)))

    return res.minimum, b_inv(res.minimizer)
end

# max_utility(ρ, μ, Σ)

mm = make_mixture(μ, Σ)

prior(z) = loglikelihood(mm, z)
joint(z) = joint_density(z, ρ, r, μ, Σ, s_prob, x, 1.0, η_j, σ_k, xbar, sigma_x)

xs = -5:0.1:5
ys = -5:0.1:5
joint_grid = [joint([y, x]) for x in xs, y in ys]
prior_grid = [prior([y, x]) for x in xs, y in ys]

p1 = contour(xs, ys, exp.(joint_grid) ./ exp(maximum(joint_grid)))
p2 = contour(xs, ys, exp.(prior_grid) ./ exp(maximum(prior_grid)))

# scatter!(p1, (f[1], f[2]), label="True payoff")
# scatter!(p1, (η_j[1], η_j[2]), label="Signal")

# scatter!(p2, (f[1], f[2]), label="True payoff")
# scatter!(p2, (η_j[1], η_j[2]), label="Signal")

plot(
    p1,
    p2
)

