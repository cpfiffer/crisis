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
using DataFrames
using Parameters

# Payoff shocks
function signal(f, Σj)
    return MvNormal(f, Σj)
end

function to_density(M)
    return exp.(M .- logsumexp(M))
    # return exp.(M) ./ exp(logsumexp(M))
    # return exp.(M) ./ exp(maximum(M))
end

function prior_mixture(params)
    @unpack μ1, Σ1, μ2, Σ2, true_s = params

    g1 = MvNormal(μ1, Σ1)
    g2 = MvNormal(μ2, Σ2)
    
    mm = MixtureModel([g1, g2], true_s)
    return mm
end 

function posterior(f, η, Σj, params)
    # Construct signal distribution
    sd = signal(f, Σj)

    # P(f) + P(η | f)
    mm = prior_mixture(params)
    return logpdf(mm, f) + logpdf(sd, η)
end

function posterior_gauss(η, μ, Σ, Σj)
    Σ_post = Symmetric(Σ - Σ*inv(Σ + Σj) * Σ)
    μ_post = μ + Σ * inv(Σ + Σj) * (η - μ)
    # μ_post = μ

    return MvNormal(μ_post, Σ_post), μ_post, Σ_post
end

function posterior_state(η, Σj, params)
    @unpack μ1, Σ1, μ2, Σ2, true_s = params
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

function analytic_posterior(f, η, Σj, params)
    states, dists = posterior_state(η, Σj, params)
    new_mixture = MixtureModel(dists, states)
    return logpdf(new_mixture, f)
end

function density_grid(η, Σj, params, bound=5, step=.1)
    mm = prior_mixture(params)
    xs = -bound:step:bound
    ys = -bound:step:bound
    return xs,
        ys,
        [posterior([x,y], η, Σj, params) for x in xs, y in ys],
        [logpdf(mm, [x,y]) for x in xs, y in ys],
        [logpdf(signal([x,y], Σj), η) for x in xs, y in ys],
        [analytic_posterior([x,y], η, Σj, params) for x in xs, y in ys]
end

function convex_combo(weight, thing1, thing2)
    return weight * thing1 + (1-weight) * thing2
end

function gmm_covar(m1, m2, s1, s2, s_hat)
    mbar = convex_combo(s_hat, m1, m2)

    t1 = (m1 - mbar) * (m1 - mbar)'
    t2 = (m2 - mbar) * (m2 - mbar)'

    sbar = convex_combo(s_hat, s1, s2) + convex_combo(s_hat, t1, t2)

    return mbar, sbar
end

function plot_post(η, Σj)
    nlines = 10
    px = 800
    py = 800
    xs, ys, post, prior, likelihood, analytic = density_grid(η, Σj, params)

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
function consumer_posterior(f, η, Σj, p, A, B, C, x, params; verbose=false, plotting=false, person=0)
    @unpack μ1, Σ1, μ2, Σ2, true_s, x_bar, Σ_x, sim_name = params

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
        prior = prior_mixture(params)
        # posterior = MixtureModel([post1, post2], [s_hat, 1-s_hat])
        posterior = MixtureModel([post1, post2], true_s)

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
        !ispath("plots/individuals/$sim_name/") && mkpath("plots/individuals/$sim_name/")
        savefig("plots/individuals/$sim_name/$person.png")
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

# function C(ρ, s_bar, Σ_bar_1, Σ_bar_2, Σ_x, Ση_bar)
#     sb = sigma_bar(s_bar, Σ_bar_1, Σ_bar_2)
#     return -ρ .* sb * (I + 1/ρ^2 * inv(Σ_x) * Ση_bar)
# end

function qstar(ρ, s_hat, Σ_h, Σ_l, μ_h, μ_l, p, r)
    # return 1/ρ * inv(true_s[1] .* Σ_h + (1 - true_s[1]) .* Σ_l) * (true_s[1] .* μ_h + (1-true_s[1]) .* μ_l)
    return 1/ρ * inv(s_hat .* Σ_h + (1 - s_hat) .* Σ_l) * (s_hat .* μ_h + (1-s_hat) .* μ_l - p .* r)
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
    return -exp(-ρ * Wj)
end

function expected_utility(p, r, ρ, s_hat, Σ_h, Σ_l, μ_h, μ_l)
    q = qstar(ρ, s_hat, Σ_h, Σ_l, μ_h, μ_l, p, r)
    μs = s_hat.*μ_h + (1-s_hat).* μ_l
    Σs = s_hat.*Σ_h + (1-s_hat).* Σ_l

    # payoff_good = -exp(-ρ * q' * (μ_h - p .* r) + only(ρ^2/2 * q'Σ_h*q))
    # payoff_bad = -exp(-ρ * q' * (μ_l - p .* r) + only(ρ^2/2 * q'Σ_l*q))
    # exp_util = s_hat * payoff_good + (1-s_hat) * payoff_bad

    t1 = only(ρ .* q' * (μs))
    t2 = -only(ρ^2/2 .* q' * Σs * q)
    t3 = -only(ρ .* q' * p .* r)
    return (uj=t1 + t2 + t3, u_ret=t1, u_var=t2, u_price=t3)
end

function investor_kl(zs)
    return missing
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
function equilibrium(params; finalplot = true)
    # Extract parameters
    @unpack μ1, Σ1, μ2, Σ2, true_s, x_bar, Σ_x, sim_name, ρ, J, K, seed, do_seed = params
    
    g1 = MvNormal(μ1, Σ1)
    g2 = MvNormal(μ2, Σ2)
    
    # Setup
    !ispath("results/individuals/$sim_name") && mkpath("results/individuals/$sim_name")

    # Set a seed
    do_seed && Random.seed!(seed)

    # Draw a state
    s = rand(Categorical(true_s))

    # Draw payoffs
    f = s == 1 ? rand(g1) : rand(g2)
    n_assets = length(f)

    # Calculate supply
    x = rand(MvNormal(x_bar, Σ_x))

    # Generate signals
    # Σj = [rand_attention(n_assets, 1) for _ in 1:J]

    # Two types
    divline = div(J, 2)
    Σj = vcat(
        repeat([diagm([inv(K*0.9), inv(K*0.1)])], divline),
        repeat([diagm([inv(K*0.1), inv(K*0.9)])], J - divline),
    )
    η = [rand(signal(f, Σj[j])) for j in 1:J]

    # General inits
    r = 1

    # Make a grid
    function init_target(θ)
        a,b,c = price_matrices(θ, n_assets)
        p = a + b*f + c*x
        return sum((p - f) .^ 2)
    end

    init_θ = vcat(
        zeros(n_assets) , 
        -vec(inv(diagm(f))), -vec(diagm(ones(n_assets))))
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
        res = store ? [] : missing

        try
            # Calculate consumer posterior
            u_sum = 0
            total_q = zeros(n_assets)
            for j in 1:J
                # Find the integral
                zs = if j == 1 || j == divline+1
                    consumer_posterior(f, η[j], Σj[j], p, a, b, c, x, params; person=j, plotting=plotting)
                else
                    consumer_posterior(f, η[j], Σj[j], p, a, b, c, x, params; person=j, plotting=false)
                end
                q = qstar(ρ, zs.s_hat, zs.Σ1, zs.Σ2, zs.μ1, zs.μ2, p, r)
                total_q += q

                # Store results if we've got em'
                if store
                    # Compute expected utility
                    uj, t1, t2, t3 = expected_utility(p, r, ρ, zs.s_hat, zs.Σ1, zs.Σ2, zs.μ1, zs.μ2)
                    u_sum += uj

                    mus, sigs = gmm_covar(zs.μ1, zs.μ2, zs.Σ1, zs.Σ2, zs.s_hat)
                    mus_backup = zs.s_hat * zs.μ1 + (1-zs.s_hat) * zs.μ2
                    @assert mus == mus_backup
                    expected_return = mus ./ p .- 1

                    push!(res, (
                        s_hat = zs.s_hat,
                        mu = mus,
                        mu_1 = zs.μ1,
                        mu_2 = zs.μ2,
                        Sigma = sigs,
                        Sigma_1 = zs.Σ1,
                        Sigma_2 = zs.Σ2,
                        forecast_error = f - mus,
                        forecast_sse = sum((f - mus).^2),
                        expected_return = expected_return,
                        quantity = q,
                        utility = uj,
                        utility_mean = t1,
                        utility_var = t2,
                        utility_price = t3,
                        signal = η[j],
                        attn_mat = diag(Σj[j]),
                        payoff = f,
                        state = s,
                        true_mu = true_s[1] * μ1 + (1-true_s[1]) * μ2,
                        true_mu_1 = μ1,
                        true_mu_2 = μ2,
                        true_sigma_1 = Σ1,
                        true_sigma_2 = Σ2,
                        price = p,
                        act_return = f ./ p,
                        exante = (f - p)' * inv(Σj[j]) * (f - p),
                        A = a,
                        B = b,
                        C = c,
                        group = j <= divline ? "attn_asset_1" : "attn_asset_2",
                    ))

                    if finalplot && j == 1 || j == divline+1
                        open("results/individuals/$sim_name/$j.txt", write=true, create=true) do io
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
                            println(io, "E[r]:               ", round.(expected_return, digits=2))
                            println(io)
                            println(io, "\nQuantity")
                            println(io, "q_j:       ", round.(q, digits=3))
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
                            println(io, "Signa1:    ", round.(Σ1, digits=3))
                            println(io, "Price:     ", round.(p, digits=3))
                            println(io, "Return:    ", round.(f ./ p, digits=3))
                            println(io, "A:         ", round.(a, digits=3))
                            println(io, "B:         ", round.(b, digits=3))
                            println(io, "C:         ", round.(c, digits=3))
                        end
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

            return norm_diff, res #+ param_norm
        catch e
            if e isa PosDefException
                return Inf, res
            end

            rethrow(e)
        end
    end

    tt(z; kwargs...) = target(z; kwargs...)[1]

    function g!(G, θ)
        G[:] = ForwardDiff.gradient(tt, θ)
    end

    # Call it
    # target(init_θ, verbose=true)
    # nltarget(init_θ)

    # Using Optim.jl
    res = optimize(
        tt,
        # g!,
        init_θ, #randn(length(init_θ)),
        # Newton(linesearch = BackTracking(order=2)),
        LBFGS(),
        autodiff=:forwarddiff,
        Optim.Options(iterations=100_000)
    )
    println("Solved!")
    display(res)

    actual = target(res.minimizer, verbose=finalplot, plotting=finalplot, store=true)
    df = DataFrame(actual[2])

    try
        if finalplot
            sz = (800, 800)
            # Do economy plotting
            !ispath("plots/economy/$sim_name/") && mkpath("plots/economy/$sim_name/")

            # Plot s_hat
            density(df.s_hat, group=df.group, title="Distribution of s_hat", size=sz)
            savefig("plots/economy/$sim_name/s_hat.png")

            # Plot forecast errors
            density(df.forecast_sse, group=df.group, title="Distribution of forecast SSE", size=sz)
            savefig("plots/economy/$sim_name/forecast_sse.png")

            # Plot expected utility
            density(df.utility, group=df.group, title="Distribution of E[uj]", size=sz)
            savefig("plots/economy/$sim_name/utility.png")

            # Plot expected utility (term 1)
            density(df.utility_mean, group=df.group, title="Distribution of E[uj], term 1 (mean payoff)", size=sz)
            savefig("plots/economy/$sim_name/utility-1.png")

            # Plot expected utility (term 2)
            density(df.utility_var, group=df.group, title="Distribution of E[uj], term 2 (variance disutility)", size=sz)
            savefig("plots/economy/$sim_name/utility-2.png")

            # Plot expected utility (term 1)
            density(df.utility_price, group=df.group, title="Distribution of E[uj], term 3 (price disutility)", size=sz)
            savefig("plots/economy/$sim_name/utility-3.png")

            # Plot ex-ante utility
            density(df.exante, title="Density of ex-ante", size=sz)
            savefig("plots/economy/$sim_name/exante.png")

            # # Plot utility function
            # density(df.exp_utility, group=df.group, title="Distribution of E[uj], exponential")
            # savefig("plots/economy/$sim_name/utility-all.png")

            # # Plot expected good utility
            # density(df.good_utility, group=df.group, title="Distribution of E[uj], good state")
            # savefig("plots/economy/$sim_name/utility-good.png")

            # # Plot expected bad utility
            # density(df.bad_utility, group=df.group, title="Distribution of E[uj], bad state")
            # savefig("plots/economy/$sim_name/utility-bad.png")

            # Plot expected returns
            plot(
                density(map(z -> z[1], df.expected_return), group=df.group, title="Distribution of expected returns (A1)"),
                density(map(z -> z[2], df.expected_return), group=df.group, title="Distribution of expected returns (A2)"),
                size=sz
            )
            scatter!([tuple(df.act_return[1]...)], size=sz)
            savefig("plots/economy/$sim_name/expected_returns.png")

            # Plot quantity
            plot(
                density(map(z -> z[1], df.quantity), group=df.group, title="Distribution of quantity (A1)", size=sz),
                density(map(z -> z[2], df.quantity), group=df.group, title="Distribution of quantity (A2)", size=sz),
            )
            savefig("plots/economy/$sim_name/quantity.png")

            # Plot ex-post mus
            scatter(map(z -> tuple(z...), df.mu), group=df.group, title="Distribution of μ_hat", size=sz)
            savefig("plots/economy/$sim_name/mu_hat.png")
        end
    catch e
        rethrow(e)
    end
    # return res
    
    # Using NLSolve.jl
    # res = nlsolve(target, init_θ, autodiff=:forward, iterations=10_000)
    # target(res.zero, verbose=true)
    # return res
    return df
end

# Set up parameters
all_j = 1_000
all_k = 10

## Baseline, nothing interesting
baseline_params = (
    sim_name = "baseline",
    μ1 = [1.0, 1.0],
    μ2 = [1.0, 1.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0; 0 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

## μ1[2] ≠ μ2[2]
two_meanshift = (
    sim_name = "a2-mean-shift",
    μ1 = [1.0, 2.0],
    μ2 = [1.0, -2.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0; 0 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

## Σ1[2,2] ≠ Σ2[2,2]
two_varshift = (
    sim_name = "a2-var-shift",
    μ1 = [1.0, 1.0],
    μ2 = [1.0, 1.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0; 0 5.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

## μ1[2] ≠ μ2[2], Σ1[2,2] ≠ Σ2[2,2]
two_meanvarshift = (
    sim_name = "a2-meanvar-shift",
    μ1 = [1.0, 2.0],
    μ2 = [1.0, -2.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0; 0 5.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

# State 2 correlation rises
morecorr = (
    sim_name = "more-corr",
    μ1 = [1.0, 1.0],
    μ2 = [1.0, 1.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0.1; 0.1 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

# State 2 correlation is negative
lesscorr = (
    sim_name = "less-corr",
    μ1 = [1.0, 1.0],
    μ2 = [1.0, 1.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 -0.1; -0.1 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

param_set = [
    baseline_params,
    two_meanshift,
    two_varshift,
    two_meanvarshift,
    morecorr,
    lesscorr,
]

# sims = map(p -> p.sim_name => equilibrium(p; finalplot=true), param_set)
# println("Done, C")
# v = equilibrium(params);

# prices = map(z -> z[1] => z[2][2][1,:price], sims)
# prc = map(z -> tuple(z[2]...), prices)
# simnames = map(z -> z[1], sims)
# scatter(prc, group=simnames, legend=:topleft, xlabel="Asset 1", ylabel="Asset 2")
# savefig("prices.png")

# Plot attention line
# ks = 1:0.5:10
# a1 = []
# a2 = []
# for k in ks
#     new_params = (
#         sim_name = "k-$k",
#         μ1 = [1.0, 2.0],
#         μ2 = [1.0, -2.0],
#         Σ1 = [2.0 0; 0 1.0],
#         Σ2 = [1.0 0.0; 0.0 1.0],
#         x_bar = [0.0, 0.0],
#         Σ_x = [1.0 0.0; 0.0 1.0],
#         J = 5_000,
#         K = k,
#         ρ = 1,
#         true_s = [0.5, 0.5],
#         seed = 1
#     )

#     eq = equilibrium(new_params)
#     push!(a1, eq[2].price[1][1])
#     push!(a2, eq[2].price[1][2])
# end

# plot(ks, a1)
# savefig("attention-a1.png")
# plot(ks, a2)
# savefig("attention-a2.png")

function exante_calc(N=1_000)
    np = (
        sim_name = "a2-mean-shift",
        μ1 = [1.0, 1.0],
        μ2 = [-1.0, 1.0],
        Σ1 = [1.0 0; 0 1.0],
        Σ2 = [1.0 0.1; 0.1 1.0],
        x_bar = [0.0, 0.0],
        Σ_x = [1.0 0.0; 0.0 1.0],
        J = 1_000,
        K = 1.0,
        ρ = 1,
        true_s = [.8,.2],
        seed = 1,
        do_seed = true
    )

    mm = prior_mixture(np)
    supply = MvNormal(np.x_bar, np.Σ_x)

    eq = equilibrium(np, finalplot=false)
    a = eq.A[1]
    b = eq.B[1]
    c = eq.C[1]

    function tgt(σ_j)
        try
            k1 = logistic(σ_j[1])
            ex = zeros(N)
            eij = diagm([k1 * np.K, (1-k1) * np.K])
            for k in 1:N
                f = rand(mm)
                x = rand(supply)
                p = a + b*f + c*x
                ex[k] = (p - f)' * inv(eij) * (p-f)
            end

            return mean(ex)
        catch e
            return Inf
        end
    end

    xs = -3:0.1:3
    sims = map(z -> tgt([z]), xs)
    # optimal = [m, 1-m]
    return map(logistic, xs), sims
end

ex = exante_calc()
