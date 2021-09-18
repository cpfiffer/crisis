using Distributions
using LinearAlgebra
using UnicodePlots
using StatsPlots

π = 0.5
μ1 = [0.5, 1.5]
μ2 = [1.5, 0.5]
σ1 = [1 0; 0 1]
σ2 = [2 0; 0 2]
ρ = 1

crra(ρ, w) = -exp(-ρ * w)
function _utility(ρ, q, μ, p, r, σ)
    sig = ρ^2 / 2 .* q'σ*q
    # display(sig)
    return -exp(-ρ * q' * (μ - p .* r) + only(sig))
end

function utility(s, ρ, q, p, r, μs, σs)
    return s * _utility(ρ, q, μs[1], p, r, σ[1]) + (1-s) * _utility(ρ, q, μ[2], p, r, σ[2])
end

μ = [μ1, μ2]
σ = [σ1, σ2]
p = [0.5, 0.5]

xs = [[z,k] for z in -5:1:5, k in -5:1:5]
ys = [utility(π, ρ, x, p, 1, μ, σ) for x in xs]
# ys = [crra(ρ, x) for x in xs]

# lineplot(xs, ys)
best, pos = findmax(ys)
@info "" best pos xs[pos]

plot(ys)