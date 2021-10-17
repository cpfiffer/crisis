include("posterior.jl")

for p in param_set
    val = eq_grid(p)
    plot_vals(p, val)
end