function test_get_eos()
    nparam = 5
    a = 2.0
    param = -a .+ 2a .* rand(nparam)
    # param = -a .* rand(nparam)
    p, e, cs2 = get_eos(param, "in/ska.table"; pmax=1e3, ncrust=200, logspacing=true)

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
end