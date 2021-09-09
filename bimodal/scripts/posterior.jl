using Distributions
using StatsPlots
using Optim
using StatsFuns
using LinearAlgebra

μ1 = [3.0, 1.0]
μ2 = [-3.0, -1.0]

Σ1 = [1.0 0; 0 1.0] 
Σ2 = [1.0 0; 0 1.0] .* 2
# Σ1 = [2.0 0; 0. 1.0]
# Σ2 = [1.0 0; 0 2.0]
Σj = [1 0.0; 0.0 1.0] .* 5
# Σj = [1 0.0; 0.0 5.0] .* 5

g1 = MvNormal(μ1, Σ1)
g2 = MvNormal(μ2, Σ2)
mm = MixtureModel([g1, g2], [0.5, 0.5])
signal = f -> MvNormal(f, Σj)

function to_density(M)
    return exp.(M) ./ exp(maximum(M))
end

function posterior(f, η)
    # Construct signal distribution
    sd = signal(f)

    # P(f) + P(η | f)
    return logpdf(mm, f) + logpdf(sd, η)
end

# function posterior_gauss(η, μ, Σ)
#     Σ_post = inv(inv(Σ) + inv(Σj))
#     μ_post = inv(Σ_post) * (inv(Σ) * μ + inv(Σj) * η)
#     # μ_post = μ
#     return MvNormal(μ_post, Σ_post), μ_post, Σ_post
# end

function posterior_gauss(η, μ, Σ)
    Σ_post = Symmetric(Σ - Σ*inv(Σ + Σj) * Σ)
    μ_post = μ + Σ * inv(Σ + Σj) * (η - μ)
    # μ_post = μ

    return MvNormal(μ_post, Σ_post), μ_post, Σ_post
end

function posterior_state(η, Σj)
    cd1, m1, S1 = posterior_gauss(η, μ1, Σ1)
    cd2, m2, S2 = posterior_gauss(η, μ2, Σ2)

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
        [posterior([x,y], η) for x in xs, y in ys],
        [logpdf(mm, [x,y]) for x in xs, y in ys],
        [logpdf(signal([x,y]), η) for x in xs, y in ys],
        [analytic_posterior([x,y], η, Σj) for x in xs, y in ys]
end

function plot_post(η, Σj)
    nlines = 10
    px = 800
    py = 800
    xs, ys, post, prior, likelihood, analytic = density_grid(η, Σj)
    # p1 = contour(xs, ys, 
    #     post |> to_density, title="Posterior", 
    #     legend=false, levels=nlines, size = (px, py))
    # scatter!(p1, (η[2], η[1]))
    p2 = contour(xs, ys, 
        likelihood |> to_density, title="Likelihood", 
        legend=false, levels=nlines, size = (px, py))
    scatter!(p2, (η[2], η[1]))
    p3 = contour(xs, ys, 
        prior |> to_density, title="Prior", 
        legend=false, levels=nlines, size = (px, py))
    scatter!(p3, (η[2], η[1]))
    p4 = contour(xs, ys, 
        analytic |> to_density, 
        legend=false, levels=nlines, dpi=80)
    scatter!(p4, (η[2], η[1]))
    p5 = bar(posterior_state(η, Σj)[1], legend=false, 
        title="State probability",
        size = (px, py), dpi=300)

    # Saving
    !isdir("plots") && mkdir("plots")
    !isdir("plots/state") && mkdir("plots/state")
    !isdir("plots/posterior") && mkdir("plots/posterior")

    savefig(p4, "plots/posterior/$(diag(Σj)).png")
    savefig(p5, "plots/state/$(diag(Σj)).png")

    
    return plot(p3, p2, p4, p5, dpi=300)
end

plot_post([0, 0], diagm([10,0.1]))
plot_post([0, 0], diagm([0.1,10]))

# for x in 1:2:10
#     for y in 1:2:10
#         plot_post([0, 0], diagm([x,y]))
#     end
# end
# N = 100000
# mm_draws = [rand(mm) for i in 1:N]
# eta_draws = [rand(signal(f)) for f in mm_draws]
# stacked = map(i -> vcat(mm_draws[i], eta_draws[i]), 1:N)
# emp_cov = cov(stacked)

# acov = zeros(4,4)
# acov[1:2, 1:2] = Σ1
# acov[3:4, 1:2] = Σ1
# acov[1:2, 3:4] = Σ1
# acov[3:4, 3:4] = Σ1 + Σj

# display(emp_cov) 
# display(acov)

# scatter(map(z -> tuple(z...), eta_draws), alpha=0.01)

# me = analytic |> to_density
# comp = post |> to_density

function expectation(xs, ys, grid)
    ex = zeros(2)
    for i in 1:length(xs)
        for j in 1:length(ys)
            val = grid[i,j] .* [xs[i], ys[j]]
            ex += val
        end
    end
    return ex ./ (length(xs) + length(ys))
end

# @info "KL" sum(comp .* (log.(comp) .- log.(me))) 
# display(expectation(xs, ys, me))
# display(expectation(xs, ys, comp))

# plot(
#     contour(xs, ys, me, legend=false),
#     contour(xs, ys, comp, legend=false)
# )
# contour(xs, ys, analytic |> to_density, legend=false)
# contour!(xs, ys, post    |> to_density, legend=false, c=:blues)


# function optimal_consumption(n)
#     function target_func(Σj)
#         xs, ys, post, prior, likelihood, analytic = density_grid(η, Σj)
#         return 
#     end
#     cs = zeroes(n)
#     for i in 1:n
#         η = rand()
#     xs, ys, post, prior, likelihood, analytic = density_grid(η, Σj)

# end