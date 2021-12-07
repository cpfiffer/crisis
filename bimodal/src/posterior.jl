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
using JLSO, Dates
using CSV
using FiniteDiff
import UnicodePlots

include("parameters.jl")
include("eq-piecewise.jl")

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

function prior_s(params, f)
    @unpack μ1, Σ1, μ2, Σ2, true_s = params

    g1 = MvNormal(μ1, Σ1)
    g2 = MvNormal(μ2, Σ2)

    prob1 = logpdf(g1, f) + log(true_s[1])
    prob2 = logpdf(g2, f) + log(true_s[2])

    s_est = prob1 - logsumexp(prob1, prob2)

    return exp(s_est)
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
        bounds = 5:0.1:15

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

        pl = plot(plot1, plot2)
        !ispath("plots/individuals/$sim_name/") && mkpath("plots/individuals/$sim_name/")
        savefig("plots/individuals/$sim_name/$person.png")
        # sleep(0.1)
    end

    return (s_hat=s_hat,
            μ1=μ_hat_1,
            Σ1=Σ_hat_1,
            μ2=μ_hat_2,
            Σ2=Σ_hat_2,
            log_s1=state1,
            log_s2=state2,
            denom=denom)
end

function pvar_grid(f, p, A, B, C, x, params; N=1000)
    @unpack K = params
    prior = pdf(prior_mixture(params), f)

    ts = []
    for θk in 0.01:0.01:0.999
        Σj = diagm([inv(θk * K), inv((1-θk) * K)])

        ev = 0
        for _ in 1:N
            η = rand(MvNormal(Σj))
            cps = consumer_posterior(f, η, Σj, p, A, B, C, x, params)
            m = convex_combo(cps.s_hat, cps.μ1, cps.μ2)
            s = convex_combo(cps.s_hat, cps.Σ1, cps.Σ2)
            val = only((m-p)'s*(m-p))
            ev += val
        end
        # ev /= N
        push!(ts, (payoff=f, attention=θk, expectation=ev, prior=prior, weighted=ev*prior))
    end
    return ts
end

function best_attention(pvar)
    evs = Dict()

    for p in pvar
        evs[p.attention] = 0
    end

    for p in pvar
        evs[p.attention] += p.weighted
    end

    nts = []
    for (k,v) in evs
        push!(nts, (attention=k, expectation=v))
    end

    return sort(DataFrame(nts), :expectation, rev=true)
end

function attention_grid(pvars)
    gg = Dict()
    ks = sort(unique(map(z -> z.attention, pvars)))
    for k in ks
        sub = filter(z -> z.attention == k, pvars)
        gg[k] = map(z -> z.weighted, sub)
    end
    return gg
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

function chernoff_distance(mu_a, mu_b, sigma_a, sigma_b, alpha=0.5)
    mu_diff = (1-alpha) .* mu_a - alpha .* mu_b
    sigma_diff = (1-alpha) .* sigma_a + alpha .* sigma_b
    return (1-alpha) * alpha / 2 * mu_diff' * inv(sigma_diff) * mu_diff .+
        1/2 * log(det(sigma_diff)) .- (log(det(sigma_a)^(1-alpha)) + log(det(sigma_b)^alpha))
end

function kl(mu_a, mu_b, sigma_a, sigma_b)
    logdet_a = log(det(sigma_a))
    logdet_b = log(det(sigma_b))
    mu_diff = mu_a - mu_b

    return 1/2 * (
        logdet_b - logdet_a + (mu_diff)' * inv(sigma_a) * mu_diff +
        tr(inv(sigma_b)*sigma_a) - length(mu_a)
    )
end

# durrieuThiranKelly_kldiv_icassp2012_R1-1.pdf
function approximate_kl(prior, posterior)
    x = rand(posterior, 10_000)
    return mean(logpdf(posterior, x) - logpdf(prior, x))
end

function gaussian_entropy(sigma)
    d = size(sigma, 1)
    return 1/2 * (log(det(sigma)) + d * log(2 * π) + d)
end

function investor_entropy(zs, params)
    s_hat, μ1_hat, Σ1_hat, μ2_hat, Σ2_hat = zs
    @unpack μ1, μ2, Σ1, Σ2, true_s = params

    c = [s_hat, 1-s_hat]
    m = [μ1_hat, μ2_hat]
    s = [Σ1_hat, Σ2_hat]

    h_upper = sum(c[i] * gaussian_entropy(s[i]) for i in 1:2)
    h_lower = copy(h_upper)
    for i in 1:2
        inner_upper = 0
        inner_lower = 0
        for j in 1:2
            kl_distance = kl(m[i], m[j], s[i], s[j])
            chernoff = chernoff_distance(m[i], m[j], s[i], s[j])
            # display(kl_distance)
            # display(chernoff)
            inner_upper += c[j] * exp(-kl_distance)
            inner_lower += c[j] * exp(-chernoff)
        end
        h_upper -= c[i] * log(inner_upper)
        h_lower -= c[i] * log(inner_lower)
    end

    return h_lower, h_upper
end

# function investor_kl(zs, params)
#     s_hat, μ1_hat, Σ1_hat, μ2_hat, Σ2_hat = zs
#     @unpack μ1, μ2, Σ1, Σ2, true_s = params

#     c = [s_hat, 1-s_hat]

#     h =
#     for i in 1:2
#         for j in 1:2
#             # h -=
#         end
#     end
# end

# The loop!
function equilibrium(
    params,
    θ,
    f,
    x;
    finalplot = true,
    store = true,
    calc_kl = false,
)
    # Extract parameters
    @unpack μ1, Σ1, μ2, Σ2, true_s, x_bar, Σ_x, sim_name, ρ, J, K, seed, do_seed = params

    g1 = MvNormal(μ1, Σ1)
    g2 = MvNormal(μ2, Σ2)

    # Setup
    !ispath("results/individuals/$sim_name") && mkpath("results/individuals/$sim_name")

    # Draw payoffs
    n_assets = length(f)

    # Generate signals
    # Σj = [rand_attention(n_assets, 1) for _ in 1:J]

    # Two types
    divline = div(J, 2)
    Σj = vcat(
        repeat([diagm([inv(K*0.5), inv(K*0.5)])], divline),
        repeat([diagm([inv(K*0.5), inv(K*0.5)])], J - divline),
    )
    η = [rand(signal(f, Σj[j])) for j in 1:J]

    # General inits
    r = 1

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
            zs = if finalplot && (j == 1 || j == divline+1)
                consumer_posterior(f, η[j], Σj[j], p, a, b, c, x, params; person=j, plotting=false)
            else
                consumer_posterior(f, η[j], Σj[j], p, a, b, c, x, params; person=j, plotting=false)
            end
            qf(p) = qstar(ρ, zs.s_hat, zs.Σ1, zs.Σ2, zs.μ1, zs.μ2, p, r)
            q = qf(p)

            total_q += q

            # Store results if we've got em'
            if store
                # Compute expected utility
                uj, t1, t2, t3 = expected_utility(p, r, ρ, zs.s_hat, zs.Σ1, zs.Σ2, zs.μ1, zs.μ2)
                u_sum += uj

                mixture_post = MixtureModel([
                    MvNormal(zs.μ1, zs.Σ1),
                    MvNormal(zs.μ2, zs.Σ2)],
                    [zs.s_hat, 1-zs.s_hat]
                )
                mus, sigs = gmm_covar(zs.μ1, zs.μ2, zs.Σ1, zs.Σ2, zs.s_hat)
                mus_backup = zs.s_hat * zs.μ1 + (1-zs.s_hat) * zs.μ2
                @assert mus == mus_backup
                expected_return = mus ./ p .- 1
                expected_variance = diag(sigs) ./ (p .^ 2)

                entropy_l, entropy_h = investor_entropy(zs, params)
                kl_calc = calc_kl ? approximate_kl(mixture_post, prior_mixture(params)) : missing

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
                    expected_variance = expected_variance,
                    quantity = q,
                    utility = uj,
                    utility_mean = t1,
                    utility_var = t2,
                    utility_price = t3,
                    signal = η[j],
                    attn_mat = diag(Σj[j]),
                    payoff = f,
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
                    entropy_lower = entropy_l,
                    entropy_upper = entropy_h,
                    kl = kl_calc
                ))

                if finalplot && (j == 1 || j == divline+1)
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
                        println(io, "H_l:                ", round(entropy_l, digits=2))
                        println(io, "H_u:                ", round(entropy_h, digits=2))
                        println(io, "kl:                 ", round(kl_calc, digits=2))
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
                        # println(io, "state:     ", s)
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

    catch e
        rethrow(e)
    end

    try
        if store
            df = DataFrame(res)
            # if finalplot
            #     sz = (500, 300)
            #     all_dpi = 180
            #     all_lines = 4
            #     # args = (size=sz, dpi=all_dpi, linewidth=all_lines)
            #     # args = (size=sz, dpi=all_dpi, linewidth=all_lines)

            #     # Do economy plotting
            #     !ispath("plots/economy/$sim_name/") && mkpath("plots/economy/$sim_name/")

            #     # Plot prior
            #     bounds = 5:0.01:15
            #     mm = MixtureModel([g1, g2], true_s)
            #     zs = [pdf(mm, [x,y]) for x in bounds, y in bounds]
            #     contour(bounds, bounds, zs, title="Prior density, $sim_name", xlabel="Asset 2", ylabel="Asset 1")
            #     savefig("plots/economy/$sim_name/prior.png")

            #     # Plot s_hat
            #     density(df.s_hat, group=df.group, title="s_hat", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/s_hat.png")

            #     # Plot forecast errors
            #     density(df.forecast_sse, group=df.group, title="Distribution of forecast SSE", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/forecast_sse.png")

            #     # Plot expected utility
            #     density(df.utility, group=df.group, title="E[uj]", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/utility.png")

            #     # Plot expected utility (term 1)
            #     density(df.utility_mean, group=df.group, title="Distribution of E[uj], term 1 (mean payoff)", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/utility-1.png")

            #     # Plot expected utility (term 2)
            #     density(df.utility_var, group=df.group, title="Distribution of E[uj], term 2 (variance disutility)", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/utility-2.png")

            #     # Plot expected utility (term 1)
            #     density(df.utility_price, group=df.group, title="Distribution of E[uj], term 3 (price disutility)", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/utility-3.png")

            #     # Plot ex-ante utility
            #     density(df.exante, title="Density of ex-ante", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/exante.png")

            #     # # Plot utility function
            #     # density(df.exp_utility, group=df.group, title="Distribution of E[uj], exponential")
            #     # savefig("plots/economy/$sim_name/utility-all.png")

            #     # # Plot expected good utility
            #     # density(df.good_utility, group=df.group, title="Distribution of E[uj], good state")
            #     # savefig("plots/economy/$sim_name/utility-good.png")

            #     # # Plot expected bad utility
            #     # density(df.bad_utility, group=df.group, title="Distribution of E[uj], bad state")
            #     # savefig("plots/economy/$sim_name/utility-bad.png")

            #     # Plot expected returns
            #     plot(
            #         density(map(z -> z[1], df.expected_return), group=df.group, title="Asset 1",size=sz, dpi=all_dpi, linewidth=all_lines),
            #         density(map(z -> z[2], df.expected_return), group=df.group, title="Asset 2",size=sz, dpi=all_dpi, linewidth=all_lines),
            #         size=sz
            #     )
            #     savefig("plots/economy/$sim_name/expected_returns.png")

            #     # Plot expected variance
            #     plot(
            #         density(map(z -> z[1], df.expected_variance), group=df.group, title="Asset 1", size=sz, dpi=all_dpi, linewidth=all_lines),
            #         density(map(z -> z[2], df.expected_variance), group=df.group, title="Asset 2", size=sz, dpi=all_dpi, linewidth=all_lines),
            #         size=sz
            #     )
            #     savefig("plots/economy/$sim_name/expected_variances.png")

            #     # Plot quantity
            #     plot(
            #         density(map(z -> z[1], df.quantity), group=df.group, title="(A1)", size=sz, dpi=all_dpi, linewidth=all_lines),
            #         density(map(z -> z[2], df.quantity), group=df.group, title="(A2)", size=sz, dpi=all_dpi, linewidth=all_lines),
            #     )
            #     savefig("plots/economy/$sim_name/quantity.png")

            #     # Plot ex-post mus
            #     scatter(map(z -> tuple(z...), df.mu), group=df.group, title="Distribution of μ_hat", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/mu_hat.png")

            #     # Entropy
            #     density(df.entropy_upper, group=df.group, title="h_u", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/entropy_u.png")

            #     density(df.entropy_lower, group=df.group, title="h_l", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/entropy_l.png")

            #     density(df.kl, group=df.group, title="kl", size=sz, dpi=all_dpi, linewidth=all_lines)
            #     savefig("plots/economy/$sim_name/kl.png")
            # end

            return df
        end
    catch e
        rethrow(e)
    end
    # return res

    # Using NLSolve.jl
    # res = nlsolve(target, init_θ, autodiff=:forward, iterations=10_000)
    # target(res.zero, verbose=true)
    # return res
    return missing
end

# The loop!
function equilibrium_grid(
    params,
    steps;
    seed = 1,
    θ_guess=nothing,
    kwargs...
)
    # Extract parameters
    @unpack μ1, Σ1, μ2, Σ2, true_s, x_bar, Σ_x, sim_name, ρ, J, K, seed, do_seed = params
    x = x_bar

    g1 = MvNormal(μ1, Σ1)
    g2 = MvNormal(μ2, Σ2)
    mm = prior_mixture(params)
    n_assets = 2

    # Setup
    !ispath("results/individuals/$sim_name") && mkpath("results/individuals/$sim_name")

    # Set a seed
    do_seed && Random.seed!(seed)

    # Generate payoff grid
    f_grid, qs = generate_grid(params, steps)

    # Two types
    divline = div(J, 2)
    Σj = vcat(
        repeat([diagm([inv(K*0.5), inv(K*0.5)])], divline),
        repeat([diagm([inv(K*0.5), inv(K*0.5)])], J - divline),
    )

    # generate centered signals
    η = [rand(MvNormal(zeros(2), Σj[j])) for j in 1:J]

    # General inits
    r = 1

    # Try to guess the initial pricing function.
    init_θ = if isnothing(θ_guess)
        vcat(
            zeros(n_assets) ,
            -vec(diagm(ones(n_assets))), -vec(diagm(ones(n_assets))))
    else
        deepcopy(θ_guess)
    end

    function target(θ; verbose=false, plotting=false, store=false)
        f = zeros(eltype(θ), 2)

        # Conjecture A, B, C matrices
        all_norm = 0
        a, b, c = price_matrices(θ, n_assets)
        # local dd
        # local dsigma
        # losses = zeros(size(f_grid))
        for ff in eachindex(f_grid)
            f[1] = f_grid[ff][1]
            f[2] = f_grid[ff][2]
            prior_weight = pdf(mm, f)

            # Determine the price at this grid point
            p = a + b*f + c*x

            # Calculate consumer posterior
            u_sum = 0
            total_q = zeros(n_assets)

            for j in 1:J
                try
                    # Find the integral
                    cps(p) = consumer_posterior(f, f + η[j], Σj[j], p, a, b, c, x, params; person=j, plotting=false)
                    cpsigma(p) = cps(p)[1]
                    zs = cps(p)

                    qf(p) = qstar(ρ, zs.s_hat, zs.Σ1, zs.Σ2, zs.μ1, zs.μ2, p, r)
                    q = qf(p)
                    # dd = ForwardDiff.value.(ForwardDiff.jacobian(qf, p))
                    # # dsigma = cpsigma(p)
                    # dsigma = FiniteDiff.finite_difference_gradient(cpsigma, p)

                    gg = map(l -> ForwardDiff.value.(cpsigma([l[1], l[2]])), f_grid)
                    display(gg)

                    total_q += q
                catch e
                    if e isa PosDefException
                        return Inf
                    end

                    rethrow(e)
                end
            end
            v = sum((total_q - x).^2)
            all_norm += v
            # losses[ff] = ForwardDiff.value(v * prior_weight)

        end
        # display(dd)
        # display(ForwardDiff.value.(dsigma))
        # println(ForwardDiff.value(all_norm))
        # # display(losses)
        return all_norm
    end

    tt(z; kwargs...) = target(z; store=false, kwargs...)

    # Call it
    # target(init_θ, verbose=true)

    # Using Optim.jl
    local result
    try
        result = optimize(
            tt,
            init_θ,
            LBFGS(),
            autodiff=:forwarddiff,
            Optim.Options(
                iterations=100_000,
            )
        )
    catch e
        if e isa InterruptException
            rethrow(e)
        end

        println(e)
        println("Retrying with backtracking line search...")
        result = optimize(
            tt,
            init_θ,
            LBFGS(linesearch = BackTracking(order=2)),
            autodiff=:forwarddiff,
            Optim.Options(iterations=100_000)
        )
    end

    # Calculate all the cool stuff now you have prices
    diffs = []
    prior = []
    disagg = []
    kls = []
    kls_var = []
    entropy_lower = []
    entropy_upper = []
    all_excess = []
    dfs = DataFrame[]
    f = zeros(n_assets)
    a,b,c = price_matrices(result.minimizer, n_assets)
    mm = prior_mixture(params)

    for ff in f_grid
        f[1] = ff[1]
        f[2] = ff[2]
        push!(prior, logpdf(mm, f))

        p = a + b*f + c*x_bar

        eq_result = equilibrium(params, result.minimizer, f, x_bar)
        try
            df = equilibrium(
                params,
                result.minimizer,
                f,
                x_bar,
                store=true,
                finalplot=false,
                calc_kl=false,
            )
            push!(diffs, f - p)

            if !ismissing(df)
                push!(disagg, sum(df.s_hat .^ 2))
                push!(kls, mean(df.kl))
                push!(kls_var, var(df.kl))
                push!(entropy_upper, mean(df.entropy_upper))
                push!(entropy_lower, mean(df.entropy_lower))
                append!(all_excess, df.forecast_error)
                push!(dfs, df)
            end
        catch e
            rethrow(e)
            if e isa InterruptException || e isa MethodError || e isa ArgumentError
                rethrow(e)
            end
            println(e)
            push!(diffs, missing)
            push!(disagg, missing)
            push!(kls, missing)
            push!(kls_var, missing)
            push!(entropy_upper, missing)
            push!(entropy_lower, missing)
        end
    end
    val = (
        rngs = qs,
        pairs=f_grid,
        diff=diffs,
        prior=prior,
        disagreement=disagg,
        kl_mean=kls,
        kl_var = kls_var,
        entropy_upper=entropy_upper,
        entropy_lower=entropy_lower,
        excess=all_excess,
    )
    CSV.write("data/individuals/$(params.sim_name).csv", vcat(dfs...))

    js_path = "jlso/$(params.sim_name)/"
    !ispath(js_path) && mkpath(js_path)
    JLSO.save(joinpath(js_path, "eq_grid-$(now()).jlso"), :values => js_path)
    return val

    # if result.minimum > 1.0
    #     result = optimize(
    #         tt,
    #         # g!,
    #         init_θ, #randn(length(init_θ)),
    #         # SimulatedAnnealing(),
    #         # ParticleSwarm(),
    #         # Newton(linesearch = BackTracking(order=2)),
    #         # LBFGS(linesearch = BackTracking(order=2)),
    #         # LBFGS(),
    #         # NewtonTrustRegion(),
    #         # NGMRES(),
    #         # autodiff=:forwarddiff,
    #         Optim.Options(iterations=100_000)
    #     )
    # end

    # # println(" result: $(result.minimum)")
    # actual = target(result.minimizer, verbose=finalplot, plotting=finalplot, store=store)
    # mats = price_matrices(result.minimizer, n_assets)
    # eq_price = mats[1] + mats[2] * f + mats[3] * x
    # df = missing
end


# sims = map(p -> p.sim_name => equilibrium(p; finalplot=true, store=true), param_set)
# println("Done, C")


# z = equilibrium(two_meanshift, store=false, finalplot=false, f=[0.5,1])
# _, (a2,b2,c2) = equilibrium(two_meanshift, store=false, finalplot=false, f=[1.0,0.5])

function generate_grid(params, steps, N=1_000_000)
    m = prior_mixture(params)
    N = 1_000_000
    draws = hcat(rand(m, N))
    dline = div(N, steps)

    mins = map(r -> minimum(sort(draws[r,:])[dline:(N-dline)]), 1:size(draws, 1))
    maxs = map(r -> maximum(sort(draws[r,:])[dline:(N-dline)]), 1:size(draws, 1))
    qs = [range(mins[i], maxs[i], length=steps) for i in 1:length(mins)]
    it = collect(Iterators.product(qs...))
    return it, qs
end

function plot_vals(params, val; num_levels=25)
    ps = "plots/params/$(params.sim_name)/"
    !ispath(ps) && mkpath(ps)

    xs = val.rngs[2]
    ys = val.rngs[1]
    # zs = map(identity, permutedims(reshape(val.prior, size(val.pairs)), [2,1]))

    savefig(contour(ys, xs, val.prior, levels=num_levels, xlabel="Asset 2", ylabel="Asset 1"), "$ps/prior.png")

    # Add grid points to prior
    contour(xs, ys, val.prior, levels=num_levels, xlabel="Asset 2", ylabel="Asset 1")
    gridpoints_x = vec(map(i -> i[2], val.pairs))
    gridpoints_y = vec(map(i -> i[1], val.pairs))
    scatter!(gridpoints_x, gridpoints_y)
    savefig("$ps/prior-grid.png")

    savefig(contour(xs, ys, val.disagreement, levels=num_levels, xlabel="Asset 2", ylabel="Asset 1"), "$ps/disagreement.png")
    savefig(contour(xs, ys, val.entropy_lower, levels=num_levels, xlabel="Asset 2", ylabel="Asset 1"), "$ps/entropy_lower.png")
    savefig(contour(xs, ys, val.entropy_upper, levels=num_levels, xlabel="Asset 2", ylabel="Asset 1"), "$ps/entropy_upper.png")
    savefig(density(map(x -> ismissing(x) ? missing : x[1], val.excess), levels=num_levels, xlabel="Asset 2", ylabel="Asset 1", linecolor = :match), "$ps/excess-1.png")
    savefig(density(map(x -> ismissing(x) ? missing : x[2], val.excess), levels=num_levels, xlabel="Asset 2", ylabel="Asset 1", linecolor = :match), "$ps/excess-2.png")
    savefig(density(map(x -> ismissing(x) ? missing : x[2], val.excess), levels=num_levels, xlabel="Asset 2", ylabel="Asset 1", linecolor = :match), "$ps/excess-2.png")
    # savefig(scatter(xs, ys, color=val.best_attention, levels=num_levels, xlabel="Asset 2", ylabel="Asset 1"), "$ps/optimal_attention.png")
    savefig(StatsPlots.histogram(val.best_attention), "$ps/optimal_attention_histogram.png")

    !ispath("$ps/attentiongrids/") && mkpath("$ps/attentiongrids/")
    for (k,v) in val.grids
        savefig(StatsPlots.contour(xs, ys, v, title = "Attention $k"), "$ps/attentiongrids/$k.png")
    end

    xvals = map(z -> z.attention, val.all_pvar)
    yvals = map(z -> z.expectation, val.all_pvar)
    points = map(z -> string(z.payoff), val.all_pvar)
    display([xvals yvals points])
    StatsPlots.plot(xvals, yvals, legend=false, group=points)
    savefig("$ps/optimality_functions.png")

    # if length(val.kl_mean) > 0
        # savefig(contour(val.rngs[2], val.rngs[1], val.kl_mean, levels=num_levels, xlabel="Asset 2", ylabel="Asset 1"), "$ps/kl_mean.png")
        # savefig(contour(val.rngs[2], val.rngs[1], val.kl_var, levels=num_levels, xlabel="Asset 2", ylabel="Asset 1"), "$ps/kl_var.png")
    # end
end


baseline_true = [
    5.893040895578876,
    6.183641522523689,
    0.39049948394149364,
    0.01785386560955143,
    0.017853865609551014,
    0.3624665268018901,
    -1.1095342102205183,
    0.26829668775657844,
    0.23892382548448576,
    -1.4985840910533748
]

for p in param_set[5:end]
    println(p.sim_name)
    val = equilibrium_grid(p, 5, θ_guess=baseline_true)
    plot_vals(p, val)
end
