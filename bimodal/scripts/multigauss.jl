using Distributions
using StatsPlots

function to_density(M)
    return exp.(M .- logsumexp(M))
    # return exp.(M) ./ exp(logsumexp(M))
    # return exp.(M) ./ exp(maximum(M))
end

function plot_norm(m, s, bound=5)
    dist = MvNormal(m, s)
    xs = -bound:0.1:bound
    ys = -bound:0.1:bound
    zs = [pdf(dist, [x, y]) for x in xs, y in ys]

    return contour(xs, ys, zs, levels=10)
end

μ = [0,0]
Σ1 = [1 0.5; 0.5 2]
Σ2 = [3 0-.5; -0.5 2] ./ 90
Σ3 = [0.9 0.1; 0.1 0.5]

d1 = MvNormal(μ, Σ1)

x = rand(d1)
# x = [1,2]

d2 = MvNormal(x, Σ2)
d3 = MvNormal(x, Σ3)

signal_2 = rand(d2)
signal_3 = rand(d2)

Σ_xx = Σ1
Σ_xη = hcat(Σ1, Σ1)
Σ_ηx = vcat(Σ1, Σ1)
Σ_ηη = [
    Σ1 + Σ2 Σ1;
    Σ1 Σ1 + Σ3;
]

signals = vcat(signal_2, signal_3)
means = vcat(μ, μ)

Σ_hat = Symmetric(Σ_xx - Σ_xη * inv(Σ_ηη) * Σ_ηx)
μ_hat = μ + Σ_xη * inv(Σ_ηη) * (signals - means)


plot(
    plot_norm(μ, Σ1),
    plot_norm(x, Σ2),
    plot_norm(x, Σ3),
    plot_norm(μ_hat, Σ_hat)
) |> display
display([x μ_hat])

@info "" pdf(MvNormal(μ_hat, Σ_hat), x)