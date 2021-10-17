function equilibrium(
    params; 
    finalplot = true, 
    store = true, 
    f=nothing, 
    s=nothing, 
    x=nothing,
    calc_kl = true,
    θ_guess = nothing,
)
    # Extract parameters
    @unpack μ1, Σ1, μ2, Σ2, true_s, x_bar, Σ_x, sim_name, ρ, J, K, seed, do_seed = params

    g1 = MvNormal(μ1, Σ1)
    g2 = MvNormal(μ2, Σ2)

    # Setup
    !ispath("results/individuals/$sim_name") && mkpath("results/individuals/$sim_name")

    # Set a seed
    do_seed && Random.seed!(seed)

    # Draw a state
    s = isnothing(s) ? rand(Categorical(true_s)) : s

    # Draw payoffs
    f = isnothing(f) ? (s == 1 ? rand(g1) : rand(g2)) : f
    n_assets = length(f)

    # Calculate supply
    x = isnothing(x) ? rand(MvNormal(x_bar, Σ_x)) : x

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

    init_θ = if isnothing(θ_guess)   
        vcat(
            zeros(n_assets) , 
            -vec(inv(diagm(f))), -vec(diagm(ones(n_assets))))
    else
        deepcopy(θ_guess)
    end
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
            for person in 1:J
                # Find the integral
                zs = if finalplot && (person == 1 || person == divline+1)
                    consumer_posterior(f, η[person], Σj[person], p, a, b, c, x, params; person=person, plotting=plotting)
                else
                    consumer_posterior(f, η[person], Σj[person], p, a, b, c, x, params; person=person, plotting=false)
                end
                q = qstar(ρ, zs.s_hat, zs.Σ1, zs.Σ2, zs.μ1, zs.μ2, p, r)
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
                    pgrid = person == 1 ? pvar_grid(f, p, a, b, c, x, params) : missing
                    # best = person == 1 ? sort(pgrid, by = x -> x.expectation) : missing
                    # optimal = person == 1 ? best_attention(pgrid) : missing

                    if person == 1
                        xvals = map(z -> z.attention, pgrid)
                        yvals = map(z -> z.expectation, pgrid)
                        # display(first(optimal, 5))
                        display(UnicodePlots.lineplot(xvals,yvals))
                    end

                    push!(res, (
                        j=person,
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
                        signal = η[person],
                        attn_mat = diag(Σj[person]),
                        payoff = f,
                        state = s,
                        true_mu = true_s[1] * μ1 + (1-true_s[1]) * μ2,
                        true_mu_1 = μ1,
                        true_mu_2 = μ2,
                        true_sigma_1 = Σ1,
                        true_sigma_2 = Σ2,
                        price = p,
                        act_return = f ./ p,
                        exante = (f - p)' * inv(Σj[person]) * (f - p),
                        A = a,
                        B = b,
                        C = c,
                        group = person <= divline ? "attn_asset_1" : "attn_asset_2",
                        entropy_lower = entropy_l,
                        entropy_upper = entropy_h,
                        kl = kl_calc,
                        pvar = pgrid,
                    ))

                    if finalplot && (person == 1 || person == divline+1)
                        open("results/individuals/$sim_name/$j.txt", write=true, create=true) do io
                            println(io, "Person $person\n")
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
                            println(io, "signal:    ", round.(η[person], digits=3))
                            println(io, "attention: ", round.(Σj[person], digits=3))
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

    tt(z; kwargs...) = target(z; store=false, kwargs...)[1]

    function g!(G, θ)
        G[:] = ForwardDiff.gradient(tt, θ)
    end

    # Call it
    # target(init_θ, verbose=true)
    # nltarget(init_θ)

    # Using Optim.jl
    local result
    try
        result = optimize(
            tt,
            init_θ,
            LBFGS(),
            autodiff=:forwarddiff,
            Optim.Options(iterations=100_000)
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

    if result.minimum > 1.0
        result = optimize(
            tt,
            # g!,
            init_θ, #randn(length(init_θ)),
            # SimulatedAnnealing(),
            # ParticleSwarm(),
            # Newton(linesearch = BackTracking(order=2)),
            # LBFGS(linesearch = BackTracking(order=2)),
            # LBFGS(),
            # NewtonTrustRegion(),
            # NGMRES(),
            # autodiff=:forwarddiff,
            Optim.Options(iterations=100_000)
        )
    end

    # println(" result: $(result.minimum)")
    actual = target(result.minimizer, verbose=finalplot, plotting=finalplot, store=store)
    mats = price_matrices(result.minimizer, n_assets)
    eq_price = mats[1] + mats[2] * f + mats[3] * x
    df = missing

    try
        if store 
            df = DataFrame(actual[2])
            if finalplot
                sz = (500, 300)
                all_dpi = 180
                all_lines = 4
                # args = (size=sz, dpi=all_dpi, linewidth=all_lines)
                # args = (size=sz, dpi=all_dpi, linewidth=all_lines)

                # Do economy plotting
                !ispath("plots/economy/$sim_name/") && mkpath("plots/economy/$sim_name/")

                # Plot prior
                bounds = 5:0.01:15
                mm = MixtureModel([g1, g2], true_s)
                zs = [pdf(mm, [x,y]) for x in bounds, y in bounds]
                contour(bounds, bounds, zs, title="Prior density, $sim_name", xlabel="Asset 2", ylabel="Asset 1")
                savefig("plots/economy/$sim_name/prior.png")

                # Plot s_hat
                density(df.s_hat, group=df.group, title="s_hat", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/s_hat.png")

                # Plot forecast errors
                density(df.forecast_sse, group=df.group, title="Distribution of forecast SSE", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/forecast_sse.png")

                # Plot expected utility
                density(df.utility, group=df.group, title="E[uj]", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/utility.png")

                # Plot expected utility (term 1)
                density(df.utility_mean, group=df.group, title="Distribution of E[uj], term 1 (mean payoff)", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/utility-1.png")

                # Plot expected utility (term 2)
                density(df.utility_var, group=df.group, title="Distribution of E[uj], term 2 (variance disutility)", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/utility-2.png")

                # Plot expected utility (term 1)
                density(df.utility_price, group=df.group, title="Distribution of E[uj], term 3 (price disutility)", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/utility-3.png")

                # Plot ex-ante utility
                density(df.exante, title="Density of ex-ante", size=sz, dpi=all_dpi, linewidth=all_lines)
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
                    density(map(z -> z[1], df.expected_return), group=df.group, title="Asset 1",size=sz, dpi=all_dpi, linewidth=all_lines),
                    density(map(z -> z[2], df.expected_return), group=df.group, title="Asset 2",size=sz, dpi=all_dpi, linewidth=all_lines),
                    size=sz
                )
                savefig("plots/economy/$sim_name/expected_returns.png")

                # Plot expected variance
                plot(
                    density(map(z -> z[1], df.expected_variance), group=df.group, title="Asset 1", size=sz, dpi=all_dpi, linewidth=all_lines),
                    density(map(z -> z[2], df.expected_variance), group=df.group, title="Asset 2", size=sz, dpi=all_dpi, linewidth=all_lines),
                    size=sz
                )
                savefig("plots/economy/$sim_name/expected_variances.png")

                # Plot quantity
                plot(
                    density(map(z -> z[1], df.quantity), group=df.group, title="(A1)", size=sz, dpi=all_dpi, linewidth=all_lines),
                    density(map(z -> z[2], df.quantity), group=df.group, title="(A2)", size=sz, dpi=all_dpi, linewidth=all_lines),
                )
                savefig("plots/economy/$sim_name/quantity.png")

                # Plot ex-post mus
                scatter(map(z -> tuple(z...), df.mu), group=df.group, title="Distribution of μ_hat", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/mu_hat.png")

                # Entropy
                density(df.entropy_upper, group=df.group, title="h_u", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/entropy_u.png")

                density(df.entropy_lower, group=df.group, title="h_l", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/entropy_l.png")

                density(df.kl, group=df.group, title="kl", size=sz, dpi=all_dpi, linewidth=all_lines)
                savefig("plots/economy/$sim_name/kl.png")
            end
        end
    catch e
        rethrow(e)
    end
    # return res

    # Using NLSolve.jl
    # res = nlsolve(target, init_θ, autodiff=:forward, iterations=10_000)
    # target(res.zero, verbose=true)
    # return res
    if store
        return df, mats, eq_price
    else
        return missing, mats, eq_price
    end
end

function eq_grid(params; steps=25)
    !ispath("data/individuals/") && mkpath("data/individuals/")

    it, qs = generate_grid(params, steps)
    m = prior_mixture(params)
    as = []
    bs = []
    cs = []
    ps = []
    diffs = []
    prior = []
    disagg = []
    kls = []
    kls_var = []
    entropy_lower = []
    entropy_upper = []
    all_excess = []
    dfs = DataFrame[]
    att = []
    all_pvar = []
    attention_grids = []

    for i in 1:size(it, 1)
        for j in 1:size(it, 2)
            println("$i $j $(it[i,j]) $(params.sim_name) ")
            ff = vcat(it[i,j]...)
            push!(prior, pdf(m, ff))
            try
                df, (a,b,c), p = equilibrium(
                    params,
                    store=true, 
                    finalplot=false,
                    f=ff,
                    x=params.x_bar,
                    calc_kl=false,
                    θ_guess = length(ps) > 0 && !ismissing(as[end]) && !ismissing(bs[end]) && !ismissing(cs[end]) ? 
                        vcat(vec(as[end]), vec(bs[end]), vec(cs[end])) :
                        nothing
                )


                if !ismissing(df)
                    # a = df.A[1]
                    # b = df.B[1]
                    # c = df.C[1]
                    # p = df.p[1]

                    push!(disagg, sum(df.s_hat .^ 2))
                    push!(kls, mean(df.kl))
                    push!(kls_var, var(df.kl))
                    push!(entropy_upper, mean(df.entropy_upper))
                    push!(entropy_lower, mean(df.entropy_lower))
                    push!(as, a)
                    push!(bs, b)
                    push!(cs, c)
                    push!(ps, p)
                    push!(diffs, ff - p)
                    append!(all_excess, df.forecast_error)

                    subonly = dropmissing(df, :pvar)
                    zz = filter(m -> m.j == 1, df)
                    if size(subonly, 1) > 0
                        best = best_attention(subonly.pvar[1])
                        push!(att, best.attention[1])
                        append!(all_pvar, subonly.pvar[1])
                    else
                        push!(best_attention, missing)
                    end

                    df[!, :i] .= i
                    df[!, :j] .= j
                    push!(dfs, df)
                end
            catch e
                # rethrow(e)
                if e isa InterruptException || e isa MethodError || e isa ArgumentError
                    rethrow(e)
                end
                println(e)
                push!(as, missing)
                push!(bs, missing)
                push!(cs, missing)
                push!(ps, missing)
                push!(diffs, missing)
                push!(disagg, missing)
                push!(kls, missing)
                push!(kls_var, missing)
                push!(entropy_upper, missing)
                push!(att, missing)
                push!(entropy_lower, missing)
            end
        end
    end

    val = (
        rngs = qs,
        pairs=collect(it),
        a=as, 
        b=bs, 
        c=cs, 
        p=ps, 
        diff=diffs, 
        prior=prior, 
        disagreement=disagg,
        kl_mean=kls,
        kl_var = kls_var,
        entropy_upper=entropy_upper,
        entropy_lower=entropy_lower,
        excess=all_excess,
        best_attention=att,
        all_pvar = all_pvar,
        verybest = best_attention(all_pvar)[1, :attention],
        grids = attention_grid(all_pvar)
    )

    @info params.sim_name val.verybest

    CSV.write("data/individuals/$(params.sim_name).csv", vcat(dfs...))

    js_path = "jlso/$(params.sim_name)/"
    !ispath(js_path) && mkpath(js_path)
    JLSO.save(joinpath(js_path, "eq_grid-$(now()).jlso"), :values => val)
    return val
end
