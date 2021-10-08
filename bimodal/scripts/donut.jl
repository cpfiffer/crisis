using Distributions
using StatsPlots

# N = 10000
# r = rand(Beta(1, 1)) # rbeta(N, 7.0, 5.0)
# theta = 2.0*pi .* rand(Uniform(0,1), N)
# a = r .* cos.(theta)
# b = r .* sin.(theta)
# # plot(a,b)

# scatter(a,b)
# savefig("donut.png")

# density(a)
# savefig("donut-1.png")

N = 10_000
draws = rand(MvNormal([0,0], [1 0.5; 0.5 1]), N)
u1 = draws[1,:]
u2 = draws[2,:]

a = 1.15
b = 0.5

x = u1 .* a
y = (u2 ./ a) + b .* (u1.^2 .+ a^2)
scatter(x, y, title="Joint")
savefig("plots/banana.png")

density(x)
density!(y, title="Marginals")
savefig("plots/banana-marginals.png")

cornerplot(hcat(x, y))
savefig("plots/banana-pair.png")

N = 10_000
m1 = [0,2]
m2 = [2,0]
s1 = [1 0.2; 0.2 1.0]
s2 = [1 -0.1; -0.1 2.0]

g1 = MvNormal(m1, s1)
g2 = MvNormal(m2, s2)
mm = MixtureModel([g1, g2])

draws = [tuple(rand(mm)...) for _ in 1:N]
scatter(draws, alpha =0.1, title="Joint")
savefig("plots/mixture.png")

x = map(z -> z[1], draws)
y = map(z -> z[2], draws)

density(x)
density!(y, title="Marginal")
savefig("plots/mixture-marginal.png")
