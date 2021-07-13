import Pkg; Pkg.activate(".")

using Distributions
using LinearAlgebra
using Statistics
using StatsPlots

D = 2
# μ = randn(D)
μ = [1.5, -5.8]
Σ = diagm([1.5, 2.0])
target = MvNormal(μ, Σ)

N = 1000
data = rand(target, N)

function niw(mu, lambda, psi, nu)
    Σ = rand(InverseGamma(nu, psi))
    y = MvNormal(mu, Σ ./ lambda)
end

function posterior_mean(data, sigma; V_0 = diagm(ones(D)), m_0 = zeros(D, 1))
    N = size(data, 2)
    xbar = mean(data, dims=2)
    V = Symmetric(inv(inv(V_0) + N .* inv(sigma)))

    # @info "" V inv(sigma) N xbar V_0 m_0
    m = V * (inv(sigma) * (N .* xbar) + inv(V_0) * m_0)

    return MvNormal(vec(m), V)
end

function posterior_variance(
    data, 
    mu; 
    S_0 = diagm(ones(D)),
    ν_0 = D+2
)
    N = size(data, 2)
    ν_N = ν_0 + N
    demeaned = data .- mean(data, dims=2)
    S_μ = (demeaned*demeaned')
    S_n = (S_0 + S_μ)

    return InverseWishart(ν_N, S_n)
end

for i in [5, 10, N]
    d = posterior_mean(data[:,1:i], Σ)
    v = posterior_variance(data[:,1:i], mean(d))
    # println(d)
    # println(mean(v))

    newdist = MvNormal(d.μ, Symmetric(mean(v)))
    println(newdist)
    # window = -10:0.1:10
    # points = [pdf(newdist, [a,b]) for a in window, b in window]
    # p = contour(points, title = string(diag(mean(v))))
    # !isdir("niw-plots") && mkpath("niw-plots")
    # savefig("niw-plots/$i.png")
end