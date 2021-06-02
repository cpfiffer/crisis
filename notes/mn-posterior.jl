using Distributions
using Parameters
using AdvancedMH

# Params
num_assets = 2
data_points = 1000

# Draw sigma
true_S0 = diagm(randn(num_assets) .^ 2)
Σ = rand(InverseWishart(num_assets + 2, true_S0))

# Draw mu
μ = rand(MvNormal(num_assets, 1))

# Set up prior distributions
μ0 = zeros(num_assets)
V0 = diagm(ones(num_assets))
μ_prior = MvNormal(μ0, V0)

S0 = diagm(ones(num_assets))
ν0 = num_assets + 2
Σ_prior = InverseWishart(ν0, S0)

# Draw the data
R = rand(MvNormal(μ, Σ), data_points)

# Define a log joint
function target(θ)
    @unpack m, V= θ
    S = S0
    
    # Handle likelihood first
    D = MvNormal(m, V)
    ℓ = mapreduce(i -> logpdf(D, R[:,i]), +, 1:size(R, 2))

    # Then handle priors
    # Mean first:
    ℓ += logpdf(μ_prior, m)
    ℓ += logpdf(Σ_prior, S)
end

true_θ = (
    m = μ,
    V = diagm(ones(num_assets)),
    S = true_S0,
)

target(true_θ)

proposal = (
    m = StaticProposal(μ_prior),
    V = StaticProposal(Σ_prior)
)

sample(
    DensityModel(target), 
    MetropolisHastings(proposal), 
    1_000; 
    chain_type=Vector{NamedTuple}
)