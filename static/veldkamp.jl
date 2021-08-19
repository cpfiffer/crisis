using Distributions
using LinearAlgebra
using Random

Random.seed!(1)

function payoffs(μf, Σz, Γ)
    zdist = MvNormal(Σz)
    z = rand(zdist)

    f = μf + Γ*z
    return f, z
end

function risk_payoffs(μf, Σz, Γ_inv)
    zdist = MvNormal(Σz)
    z = rand(zdist)

    f = Γ_inv * μf + z
    return f, z
end

function risk_supply(μx, σx)
    xdist = MvNormal(diagm(σx))
    x = rand(xdist) + μx
    return x
end

function attention_weights(γ)
    d = Dirichlet(γ)
    return rand(d)
end

function signals(z, K, α)
    noise_dist = MvNormal(inv(diagm(α .* K)))
    return rand(noise_dist) + z
end

function wealth(q, f, p; W0=1, r=1)
    return r*W0 + q' * (f - p .* r)
end

function utility_t2(ew, vw; ρ=1)
    return ρ*ew - (ρ^2 / 2) * vw
end

q = [0.5, 0.5]
f = [1.0, 2.0]
p = [0.1, 0.1]
μf = [1.0, 2.0]
Σz = diagm([0.1, 0.25])
μx = [0.1, 10.]
σx = [0.2, 3.0]
K = 1
Γ = [1.0 0.05; 0.0 1.0]
Γ_inv = inv(Γ)

f, z = risk_payoffs(μf, Σz, Γ_inv)

γ = [2, 1]
α = attention_weights(γ)

@info "" wealth(q, f, p) utility_t2(1, 0.2) risk_supply(μx, σx) γ α f z
@info "" signals(z, K, α)
