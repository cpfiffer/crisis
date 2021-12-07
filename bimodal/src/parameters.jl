# Set up parameters
all_j = 1000
all_k = 1

## Baseline, nothing interesting
baseline_params = (
    sim_name = "baseline",
    μ1 = [10.0, 10.0],
    μ2 = [10.0, 10.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0; 0 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

## μ1[2] ≠ μ2[2]
two_meanshift = (
    sim_name = "a2-mean-shift",
    μ1 = [10.0, 12.0],
    μ2 = [10.0, 8.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0; 0 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

similar_assets = (
    sim_name = "means-only",
    μ1 = [8.0, 12.0],
    μ2 = [8.0, 12.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0; 0 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

## μ1[2] ≠ μ2[2]
corr_test = (
    sim_name = "corr-mean-shift",
    μ1 = [10.0, 12.0],
    μ2 = [10.0, 8.0],
    Σ1 = [1.0 0.1; 0.1 1.0],
    Σ2 = [1.0 0.1; 0.1 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)


## Σ1[2,2] ≠ Σ2[2,2]
two_varshift = (
    sim_name = "a2-var-shift",
    μ1 = [10.0, 10.0],
    μ2 = [10.0, 10.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0; 0 5.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

## μ1[2] ≠ μ2[2], Σ1[2,2] ≠ Σ2[2,2]
two_meanvarshift = (
    sim_name = "a2-meanvar-shift",
    μ1 = [10.0, 12.0],
    μ2 = [10.0, 8.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0; 0 5.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

# State 2 correlation rises
morecorr = (
    sim_name = "more-corr",
    μ1 = [10.0, 10.0],
    μ2 = [10.0, 10.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0.2; 0.2 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

# State 2 correlation is negative
lesscorr = (
    sim_name = "less-corr",
    μ1 = [10.0, 10.0],
    μ2 = [10.0, 10.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 -0.2; -0.2 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

# State 2 correlation rises, means shift
morecorr_mean = (
    sim_name = "more-corr-meanshift",
    μ1 = [10.0, 12.0],
    μ2 = [10.0, 8.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [1.0 0.2; 0.2 1.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

# State 2 correlation rises, variances & means shift
morecorr_meanvar = (
    sim_name = "more-corr-meanvarshift",
    μ1 = [10.0, 12.0],
    μ2 = [10.0, 8.0],
    Σ1 = [1.0 0; 0 1.0],
    Σ2 = [2.0 0.4; 0.4 2.0],
    x_bar = [0.0, 0.0],
    Σ_x = [1.0 0.0; 0.0 1.0],
    J = all_j,
    K = all_k,
    ρ = 1,
    true_s = [0.5, 0.5],
    seed = 1,
    do_seed = true
)

param_set = [
    baseline_params,
    two_meanshift,
    similar_assets,
    # two_varshift,
    # corr_test,
    two_meanvarshift,
    # morecorr,
    # lesscorr,
    # morecorr_mean,
    morecorr_meanvar
]

function plookup(s)
    sub = filter(x -> x.sim_name == s, param_set)
    if length(sub) == 1
        return only(sub)
    else
        return missing
    end
end