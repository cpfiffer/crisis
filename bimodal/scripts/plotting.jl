function plot_mixture(μ, Σ)
    mm = MixtureModel([MvNormal(μ[i], Σ[i]) for i in 1:n_states])

    xs = -5:0.1:5
    ys = -5:0.1:5
    zs = [pdf(mm, [x,y]) for x in xs, y in ys]

    contour(xs, ys, zs, title="Mixture model density, 13 states")
    savefig("writing/plots/mixture-1.png")
end