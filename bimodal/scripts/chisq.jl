using Distributions, StatsPlots

d1 = NoncentralChisq(10, 5)
d2 = NoncentralChisq(20, 5)
d3 = NoncentralChisq(15, 5)
m = MixtureModel([d1, d2])

xs = 0.01:0.1:15
ys1 = map(z -> pdf(d1, z), xs)
ys2 = map(z -> pdf(d2, z), xs)
ys3 = map(z -> pdf(m, z), xs)
ys4 = map(z -> pdf(d3, z), xs)

plot(xs, ys1)
plot!(xs, ys2)
plot!(xs, ys3)
plot!(xs, ys4)
savefig("plots/chisq-mixture.png")