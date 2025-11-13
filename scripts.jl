function test_get_eos()
    nparam = 11
    a = 5.0
    # a = 0.5
    param = -a .+ 2a .* rand(nparam)
    # param = -a .* rand(nparam)
    pmax = 700.0
    p, e, cs2 = get_eos(param, "in/ska.table"; pmax=pmax, ncrust=200, logspacing=true, pa_eos=true)

    fig = Figure()

    ax1 = Axis(fig[1, 1]; 
        xlabel="e", ylabel="p",
        xscale=log10, yscale=log10,
        # limits=(0.7, e[end]/e0, p[findfirst(x -> x > e0, e)], p[end]),
    )
    
    lines!(ax1, e, p; color=:blue, label="EOS")

    ax2 = Axis(fig[1, 2]; 
        xlabel="e", ylabel="cs2",
        xscale=log10,
        # limits=(0.01, e[end]/e0, 0, 1.1*maximum(cs2)),
    )

    lines!(ax2, e, cs2; color=:green, label="cs2 from Gamma")

    save("out/eos_test.png", fig)

    pc_start = 0.1
    pc_end = pmax
    nstars = 100

    pc, M, R, Lambda = getMRLambdadiagram(p, e, pc_start, pc_end, nstars)

    fig2 = Figure()
    ax3 = Axis(fig2[1, 1]; 
        xlabel="R (km)", ylabel="M (M☉)",
        xscale=identity, yscale=identity,
    )

    lines!(ax3, R, M; color=:red, label="M-R")

    save("out/mr_test.png", fig2)

    fig3 = Figure()
    ax4 = Axis(fig3[1, 1];
        xlabel="M (M☉)", ylabel="Λ",
        xscale=identity, yscale=log10,
        limits=(0, maximum(M), 1e1, 1e5),
    )

    lines!(ax4, M, abs.(Lambda); color=:purple, label="M-Λ")

    save("out/mlambda_test.png", fig3)
end