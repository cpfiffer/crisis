using Distributions
using StatsPlots
using Optim
using StatsFuns
using LinearAlgebra
using QuadGK
using HCubature

μ1 = [1.0, 1.0]
μ2 = [-1.0, -1.0]

Σ1 = [1.0 0; 0 1.0] 
Σ2 = [1.0 0; 0 1.0] .* 2
# Σ1 = [2.0 0; 0. 1.0]
# Σ2 = [1.0 0; 0 2.0]
Σj = [1 0.0; 0.0 1.0] .* 5
# Σj = [1 0.0; 0.0 5.0] .* 5

# Supply settings
x_bar = [1.0, 1.0]
Σx = [1.0 0.0; 0.0 1.0]
supply_dist = MvNormal(x_bar, Σx)

#  Payoff shocks
g1 = MvNormal(μ1, Σ1)
g2 = MvNormal(μ2, Σ2)

mm = MixtureModel([g1, g2], [0.5, 0.5])
signal(f, Σj) = MvNormal(f, Σj)

function to_density(M)
    return exp.(M .- logsumexp(M))
    # return exp.(M) ./ exp(logsumexp(M))
    # return exp.(M) ./ exp(maximum(M))
end

function posterior(f, η, Σj)
    # Construct signal distribution
    sd = signal(f, Σj)

    # P(f) + P(η | f)
    return logpdf(mm, f) + logpdf(sd, η)
end

function posterior_gauss(η, μ, Σ, Σj)
    Σ_post = Symmetric(Σ - Σ*inv(Σ + Σj) * Σ)
    μ_post = μ + Σ * inv(Σ + Σj) * (η - μ)
    # μ_post = μ

    return MvNormal(μ_post, Σ_post), μ_post, Σ_post
end

function posterior_state(η, Σj)
    cd1, m1, S1 = posterior_gauss(η, μ1, Σ1, Σj)
    cd2, m2, S2 = posterior_gauss(η, μ2, Σ2, Σj)

    # Marginal distributions for η
    d1 = MvNormal(μ1, Σ1 + Σj)
    d2 = MvNormal(μ2, Σ2 + Σj)

    # Probabilities
    state1 = logpdf(d1, η) + log(0.5)
    state2 = logpdf(d2, η) + log(0.5)
    denom = logsumexp(state1, state2)

    x = exp(state1 - denom)
    return [x, 1-x], [cd1, cd2]
end

function analytic_posterior(f, η, Σj)
    states, dists = posterior_state(η, Σj)
    new_mixture = MixtureModel(dists, states)
    return logpdf(new_mixture, f)
end

function density_grid(η, Σj, bound=5, step=.1)
    xs = -bound:step:bound
    ys = -bound:step:bound
    return xs, 
        ys, 
        [posterior([x,y], η, Σj) for x in xs, y in ys],
        [logpdf(mm, [x,y]) for x in xs, y in ys],
        [logpdf(signal([x,y], Σj), η) for x in xs, y in ys],
        [analytic_posterior([x,y], η, Σj) for x in xs, y in ys]
end

function plot_post(η, Σj)
    nlines = 10
    px = 800
    py = 800
    xs, ys, post, prior, likelihood, analytic = density_grid(η, Σj)

    # To density
    post = to_density(post)
    prior = to_density(prior)
    likelihood = to_density(likelihood)
    analytic = to_density(analytic)

    p1 = contour(xs, ys, 
        post, title="Posterior", 
        legend=false, levels=nlines, size = (px, py))
    scatter!(p1, (η[2], η[1]))
    p2 = contour(xs, ys, 
        likelihood, title="Likelihood", 
        legend=false, levels=nlines, size = (px, py))
    scatter!(p2, (η[2], η[1]))
    p3 = contour(xs, ys, 
        prior, title="Prior", 
        legend=false, levels=nlines, size = (px, py))
    scatter!(p3, (η[2], η[1]))
    p4 = contour(xs, ys, 
        analytic, 
        legend=false, levels=nlines, dpi=80)
    scatter!(p4, (η[2], η[1]))
    p5 = bar(posterior_state(η, Σj)[1], legend=false, 
        title="State probability",
        size = (px, py), dpi=300)

    p_all = plot(p1, p3, p2, p4, p5, dpi=300)

    # Saving
    !isdir("plots") && mkdir("plots")
    !isdir("plots/state") && mkdir("plots/state")
    !isdir("plots/posterior") && mkdir("plots/posterior")
    !isdir("plots/all") && mkdir("plots/all")

    savefig(p4, "plots/posterior/$(diag(Σj)).png")
    savefig(p5, "plots/state/$(diag(Σj)).png")
    savefig(p_all, "plots/all/$(diag(Σj)).png")

    return p_all
end

# Enable to generate plots in figure 2 (as of September 10th, 2021)
# plot_post([0, 0], diagm([10,1]))
# plot_post([0, 0], diagm([1,10]))
# plot_post([0, 0], diagm([1,1]))

# Utility
function utility(f, p, q; r=1.0, W0 = 1.0, ρ = 1)
    Wj = r*W0 + q'(f - p .*r)
    return exp(-ρ * Wj)
end

# Optimum quantity
function q_star(p)
    # 
end

# The loop!
function equilibrium(Σj, J = 100)
    # Draw a state
    s = rand(Categorical([0.5, 0.5]))

    # Draw payoffs
    f = s == 1 ? rand(g1) : rand(g2)
    n_assets = length(f)

    # Calculate supply
    x = rand(MvNormal(x_bar, Σx))

    # Generate signals
    η = [rand(signal(f, Σj)) for _ in 1:J]

    # Conjecture A, B, C matrices
    A = zeros(n_assets)
    B = [diagm(ones(n_assets)) for _ in 1:J]
    C = diagm(ones(n_assets))

    p = A + sum(B[j] * η[j] for j in 1:J) + C * x
    println("Price conjecture $p")
    println("Payoffs          $f")

    # for i in 1:1000
    #     # Calculate optimum quantity
    # end
end

equilibrium()
