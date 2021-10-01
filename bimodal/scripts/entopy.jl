using Distributions, LinearAlgebra, MultivariateStats

k = 50
N = 100_000

m1 = randn(k)
m2 = randn(k) .* 2

iw = InverseWishart(k + 2, diagm(ones(k)))
s1 = rand(iw)
s2 = rand(iw) .* 1/2

g1 = MvNormal(m1, s1)
g2 = MvNormal(m2, s2)

vs = Float64[]
for weight in 0:0.01:1
    mixture = MixtureModel([g1, g2], [weight, 1-weight])

    draws = rand(mixture, N)
    p = fit(PCA, draws)
    vars = principalvars(p)
    push!(vs, vars[1])
end
