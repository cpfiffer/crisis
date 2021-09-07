# illustrates a simple choice model
using Turing
using StatsPlots
using LinearAlgebra

@model function mixture(data)
    s ~ Categorical([0.5, 0.5])

    if s == 1
        μ = [1.0, 1.0]
        Σ = [1.0 0.0; 0.0 1.0]

        # μ ~ MvNormal([1, 1], 1)
        # Σ ~ InverseWishart(6, diagm(ones(2)))
    elseif s == 2
        μ = [1.5, 0.5]
        Σ = [1.0 0.25; 0.25 1.0]

        μ = [1.5, 0.5]
        # μ ~ MvNormal([1.5, 0.5], 1)
        # Σ ~ InverseWishart(6, diagm(ones(2)) .* 2)
    end

    # data ~ Normal(μ, 1.0)
    data ~ MvNormal(μ, Σ)
end

model = mixture([1.5, 0.5])
# sampler = MH()
sampler = PG(100)
chain = sample(model, sampler, 1_000)
plot(chain)