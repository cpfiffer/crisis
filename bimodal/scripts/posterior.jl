using Distributions
using StatsPlots
using Optim
using StatsFuns
using LinearAlgebra
using QuadGK
using HCubature
using Random
using ForwardDiff
using NLsolve

# true_s = [1.0, 0.0] # Degenerate
true_s = [0.5, 0.5]
μ1 = [1.0, 1.0]
μ2 = [-1.0, -1.0]

Σ1 = [1.0 0; 0 1.0]
Σ2 = [1.0 0; 0 1.0] .* 1
# Σ1 = [2.0 0; 0. 1.0]
# Σ2 = [1.0 0; 0 2.0]
Σj = [1 0.0; 0.0 1.0] .* 5
# Σj = [1 0.0; 0.0 5.0] .* 5

# Supply settings
x_bar = [0.0, 0.0]
Σx = [1.0 0.0; 0.0 1.0]
supply_dist = MvNormal(x_bar, Σx)

#  Payoff shocks
g1 = MvNormal(μ1, Σ1)
g2 = MvNormal(μ2, Σ2)

mm = MixtureModel([g1, g2], true_s)
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
    state1 = logpdf(d1, η) + log(true_s[1])
    state2 = logpdf(d2, η) + log(true_s[2])
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


# Individual posterior
function consumer_posterior(f, η, Σj, p, A, B, C, x_bar, x, Σ_x; verbose=false, plotting=false, person=0)
    # First posterior
    S_11 = Σ1
    S_12 = [B*Σ1 Σ1]
    S_21 = S_12'
    S_22 = [B*Σ1*B' + C*Σ_x*C' B*Σ1; B*Σ1 Σ1 + Σj]

    means = vcat(A + B*μ1 + C*x_bar, μ1)
    obs = vcat(p, η)

    Σ_hat_1 = Symmetric(S_11 - S_12 * inv(S_22) * S_21)
    μ_hat_1 = μ1 + S_12 * inv(S_22) * (obs - means)

    # display(Σ_hat_1)
    # display(S_11)
    # display(S_12)
    # display(S_22)
    post1 = MvNormal(μ_hat_1, Σ_hat_1)

    # Second posterior
    S_11 = Σ2
    S_12 = [B*Σ2 Σ2]
    S_21 = S_12'
    S_22 = [B*Σ2*B' + C*Σ_x*C' B*Σ2; B*Σ2 Σ2 + Σj]

    means = vcat(A + B*μ2 + C*x_bar, μ2)
    obs = vcat(p, η)

    Σ_hat_2 = Symmetric(S_11 - S_12 * inv(S_22) * S_21)
    μ_hat_2 = μ2 + S_12 * inv(S_22) * (obs - means)
    post2 = MvNormal(μ_hat_2, Σ_hat_2)

    # Posterior states
    d1 = MvNormal(μ1, Σ1 + Σj)
    d2 = MvNormal(μ2, Σ2 + Σj)
    price_1 = MvNormal(A + B*f + C*x, Symmetric(B*Σ1*B' + C*Σ_x*C'))
    price_2 = MvNormal(A + B*f + C*x, Symmetric(B*Σ2*B' + C*Σ_x*C'))

    state1 = logpdf(d1, η) + logpdf(price_1, p) + log(true_s[1])
    state2 = logpdf(d2, η) + logpdf(price_2, p) + log(true_s[2])
    denom = logsumexp(state1, state2)
    s_hat = exp(state1 - denom)

    if verbose
        println("Consumer posterior:")
        println("\tμ1:       $μ1")
        println("\tμ_hat_1:  $μ_hat_1")
        println("\tμ2:       $μ2")
        println("\tμ_hat_2:  $μ_hat_2")
        println("\tΣ1:       $Σ1")
        println("\tΣ_hat_1:  $Σ_hat_1")
        println("\tΣ2:       $Σ2")
        println("\tΣ_hat_2:  $Σ_hat_2")
        println("\tAttention: $Σj")
        println("\ts_hat:    $s_hat")
    end

    if plotting
        # Mixtures
        prior = MixtureModel([g1, g2], true_s)
        posterior = MixtureModel([post1, post2], [s_hat, 1-s_hat])

        # Plotting variables
        bounds = -5:0.1:5

        # Contour values
        z_prior = [pdf(prior, [x,y]) for x in bounds, y in bounds]
        z_post = [pdf(posterior, [x,y]) for x in bounds, y in bounds]

        # Contour plots
        plot1 = contour(bounds, bounds, z_prior, title="Prior")
        plot2 = contour(bounds, bounds, z_post, title="Posterior")

        for subplot in [plot1, plot2]
            scatter!(subplot, [(f[2], f[1])], label="Payoffs")
            scatter!(subplot, [(p[2], p[1])], label="Price")
            scatter!(subplot, [(η[2], η[1])], label="Personal signal", legend=false)
        end

        pl = plot(plot1, plot2, dpi=180)
        !ispath("plots/individuals/") && mkpath("plots/individuals/")
        savefig("plots/individuals/$person.png")
        # sleep(0.1)
    end

    return (s_hat=s_hat, μ1=μ_hat_1, Σ1=Σ_hat_1, μ2=μ_hat_2, Σ2=Σ_hat_2)
end

# function sigma_bar(s_bar, Σ_bar_1, Σ_bar_2)
#     return (s_bar .* Σ_bar_1 + (1-s_bar).*Σ_bar_2)
# end

# function A(μ1, μ2, ρ, s_bar, Σ_bar_1, Σ_bar_2, x_bar)
#     sb = sigma_bar(s_bar, Σ_bar_1, Σ_bar_2)
#     return s_bar .* μ1 + (1-s_bar) .* μ2 - ρ .* sb * x_bar
# end

# function B(s_bar, Σ_bar_1, Σ_bar_2)
#     sb = sigma_bar(s_bar, Σ_bar_1, Σ_bar_2)
#     return I - sb * inv(2 .* inv(Σ1) + 2 .* inv(Σ2))
# end

# function C(ρ, s_bar, Σ_bar_1, Σ_bar_2, Σx, Ση_bar)
#     sb = sigma_bar(s_bar, Σ_bar_1, Σ_bar_2)
#     return -ρ .* sb * (I + 1/ρ^2 * inv(Σx) * Ση_bar)
# end

function qstar(ρ, s_hat, Σ_h, Σ_l, μ_h, μ_l)
    return 1/ρ * inv(s_hat .* Σ_h + (1 - s_hat) .* Σ_l) * (s_hat .* μ_h + (1-s_hat) .* μ_l)
end

# One b for each person
# function chop(body, amount)
#     if length(body) == amount
#         return [body]
#     else
#         return vcat([body[1:amount]], chop(body[amount+1:end], amount))
#     end
# end

# function price_matrices(θ, n_assets, J)
#     offset = n_assets^2
#     a = reshape(θ[1:n_assets], n_assets)

#     chopped = chop(θ[n_assets+1:end], offset)

#     b = chopped[1:J]
#     c = chopped[end]
#     return a, b, c
# end

# Single B matrix
function price_matrices(θ, n_assets)
    offset = n_assets^2
    a = reshape(θ[1:n_assets], n_assets)
    b = reshape(θ[n_assets+1:n_assets+offset], n_assets, n_assets)
    c = reshape(θ[n_assets+1+offset:end], n_assets, n_assets)
    return a, b, c
end

# random precision matrix
function rand_attention(n_assets, K)
    # Fraction
    γ = rand(Dirichlet(n_assets, 0.5))
    return diagm([inv(γ[i] * K) for i in 1:n_assets])
end

# Utility
function utility(f, p, q; r=1.0, W0 = 1.0, ρ = 1)
    Wj = r*W0 + q'(f - p .*r)
    return exp(-ρ * Wj)
end

function expected_utility(p, r, ρ, s_hat, Σ_h, Σ_l, μ_h, μ_l)
    q = qstar(ρ, s_hat, Σ_h, Σ_l, μ_h, μ_l)
    μs = s_hat.*μ_h + (1-s_hat).* μ_l
    Σs = s_hat.*Σ_h + (1-s_hat).* Σ_l
    return only(ρ .* q' * (μs) - ρ^2/2 .* q' * Σs * q .- ρ .* q'p .*r)
end

function investor_kl(zs)
    s_hat, μ1_hat, Σ1_hat, μ2_hat, Σ2_hat = zs
    
    # Mixtures
    prior = MixtureModel([g1, g2], true_s)
    posterior = MixtureModel([
        MvNormal(μ1_hat, Σ1_hat),
        MvNormal(μ2_hat, Σ2_hat),
    ], [s_hat, 1-s_hat])

    # Approximate KL
    rn = -5:0.01:5
    return sum(pdf(prior, [x,y]) * (logpdf(prior, [x,y]) - logpdf(posterior, [x,y])) for x in rn, y in rn)
end

# The loop!
function equilibrium(ρ=0.5, J = 2, K=100)
    # Setup
    xs = -5:1:5
    ys = -5:1:5
    !ispath("results/individuals/") && mkpath("results/individuals/")


    # Set a seed
    Random.seed!(2)

    # Draw a state
    s = rand(Categorical([0.5, 0.5]))

    # Draw payoffs
    # f = s == 1 ? rand(g1) : rand(g2)
    f = [0,0]
    n_assets = length(f)

    # Calculate supply
    x = rand(MvNormal(x_bar, Σx))

    # Generate signals
    # Σj = [rand_attention(n_assets, 1) for _ in 1:J]
    Σj = [
        diagm([inv(K*0.99), inv(K*0.01)]),
        diagm([inv(K*0.01), inv(K*0.99)])
    ]
    # η = [rand(signal(f, Σj[j])) for j in 1:J]
    η = [[0,0], [0,0]]

    # General inits
    ρ = 1
    r = 1

    # Make a grid
    function init_target(θ)
        a,b,c = price_matrices(θ, n_assets)
        p = a + b*f + c*x
        return sum((p - f) .^ 2)
    end

    init_θ = vcat(ones(n_assets) , vec(diagm(ones(n_assets))), -vec(diagm(ones(n_assets))))
    # init_θ = vcat(
    #     true_s[1] .* μ1 + true_s[2] .* μ2,
    #     vec(diagm([0.5, 0.5])),
    #     vec(diagm([0.05, 0.05]))
    # )

    # init_θ = optimize(init_target, init_θ, iterations=10_000, autodiff = :forward)
    # display(init_θ)
    # init_θ = init_θ.minimizer

    function target(θ; verbose=false, plotting=false, store=false)
        # Conjecture A, B, C matrices
        a, b, c = price_matrices(θ, n_assets)
        p = a + b*f + c*x

        try
            # Calculate consumer posterior
            u_sum = 0
            total_q = zeros(n_assets)
            for j in 1:J
                # Find the integral
                zs = consumer_posterior(f, η[j], Σj[j], p, a, b, c, x_bar, x, Σx; person=j, plotting=plotting)
                q = qstar(ρ, zs.s_hat, zs.Σ1, zs.Σ2, zs.μ1, zs.μ2)
                total_q += q

                # Compute expected utility
                uj = expected_utility(p, r, ρ, zs.s_hat, zs.Σ1, zs.Σ2, zs.μ1, zs.μ2)
                u_sum += uj

                # Store results if we've got em'
                if store
                    mus = zs.s_hat * zs.μ1 + (1-zs.s_hat) * zs.μ2
                    open("results/individuals/$j.txt", write=true, create=true) do io
                        println(io, "Person $j\n")
                        println(io, "--------------------------------------------")
                        println(io, "\nPosterior beliefs")
                        println(io, "s_hat:              ", round.(zs.s_hat, digits=3))
                        println(io, "mu:                 ", round.(mus, digits=3))
                        println(io, "mu_1:               ", round.(zs.μ1, digits=3))
                        println(io, "mu_2:               ", round.(zs.μ2, digits=3))
                        println(io, "Sigma_1:            ", round.(zs.Σ1, digits=3))
                        println(io, "Sigma_2:            ", round.(zs.Σ2, digits=3))
                        println(io, "forecast error:     ", round.(f - mus, digits=3))
                        println(io, "forecast SSE:       ", sum((f - mus).^2))
                        println(io, "KL:                 ", round(investor_kl(zs), digits=2))
                        println(io)
                        println(io, "\nUtility")
                        println(io, "u_j:       ", round.(uj, digits=3))
                        println(io)
                        println(io, "\nAttention")
                        println(io, "signal:    ", round.(η[j], digits=3))
                        println(io, "attention: ", round.(Σj[j], digits=3))
                        println(io)
                        println(io, "\nGround truth")
                        println(io, "payoffs:   ", round.(f, digits=3))
                        println(io, "state:     ", s)
                        println(io, "mu:        ", round.(true_s[1] * μ1 + (1-true_s[1]) * μ2, digits=3))
                        println(io, "mu1:       ", round.(μ1, digits=3))
                        println(io, "mu2:       ", round.(μ2, digits=3))
                        println(io, "Sigma1:    ", round.(Σ1, digits=3))
                        println(io, "Sigma2:    ", round.(Σ2, digits=3))
                        # println(io, "mu_1:               ", round.(zs.μ1, digits=3))
                        # println(io, "mu_2:               ", round.(zs.μ2, digits=3))
                        # println(io, "Sigma_1:            ", round.(zs.Σ1, digits=3))
                        # println(io, "Sigma_2:            ", round.(zs.Σ2, digits=3))
                    end
                end
            end

            norm_diff = sum((total_q - x).^2)
            param_norm = sum(θ.^2) ./ 10

            if verbose
                println("\nTarget:")
                println("\tPrice conjecture $p")
                println("\tPayoffs          $f")
                println("\tTotal q          $total_q")
                println("\tTotal x          $x")
                println("\tNorm             $norm_diff")
                println("\tParam norm       $param_norm")
                println("\ta                $a")
                println("\tb                $b")
                println("\tc                $c")
                println("\ts                $s\n")
            end

            return norm_diff #+ param_norm
        catch e
            if e isa PosDefException
                return Inf
            end

            rethrow(e)
        end
    end

    function g!(G, θ)
        G[:] = ForwardDiff.gradient(target, θ)
    end

    # Call it
    # target(init_θ, verbose=true)
    # nltarget(init_θ)

    # Using Optim.jl
    res = optimize(
        target,
        # g!,
        init_θ, #randn(length(init_θ)),
        # LBFGS(linesearch = BackTracking(order=2)),
        # autodiff=:forwarddiff,
        Optim.Options(iterations=10_000)
    )
    target(res.minimizer, verbose=true, plotting=true, store=true)
    return res

    # Using NLSolve.jl
    # res = nlsolve(target, init_θ, autodiff=:forward, iterations=10_000)
    # target(res.zero, verbose=true)
    # return res
end

v = equilibrium()
