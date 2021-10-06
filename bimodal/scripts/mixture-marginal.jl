using Distributions
using QuadGK
using StatsPlots

d1 = Normal(1, 2)
d2 = Normal(-1, 0.5)

weights = [0.75, 0.25]

mm = MixtureModel([d1, d2], weights)

target(x) = pdf(mm, x)

xs = -5:0.01:5
ys = map(target, xs)
plot(xs, ys)

# Simulate categorical/normal
function simulate(d1, d2, weights, n)
    vals = zeros(n)
    cats = rand(Categorical(weights), n)

    for i in 1:n
        if cats[i] == 1
            vals[i] = rand(d1)
        else
            vals[i] = rand(d2)
        end
    end

    return vals
end

draws = simulate(d1, d2, weights, 10000)
density!(draws)
