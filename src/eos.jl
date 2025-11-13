# Loads crust file following CompOSE convention with 3 custom output columns: p, cs2, e
function load_crust(file)
    data = readdlm(file)

    p = data[:, 4]
    cs2 = data[:, 5]
    e = data[:, 6]

    return p, e, cs2
end

function GammaCheb(p0, p, pmax, Gamma0, param::AbstractVector)
    y = -1.0 + 2.0*log(p/p0) * log(pmax/p0)^(-1)
    param_new = copy(param)
    param_new[1] = 2param[1] # for some reason chepolsum halves the first coefficient
    return Gamma0*exp(SpecialFunctions.chepolsum(y, param_new)*(1 + y))
end

function GammaPA(p0, p, pmax, Gamma0, param::AbstractVector, p_interface::AbstractVector, e_interface::AbstractVector, Gamma_interface::AbstractVector)
    idx = searchsortedlast(p_interface, p)
    if idx == length(p_interface)
        idx -= 1
    end

    Gamma = Gamma_interface[idx]*(p/p_interface[idx])^(param[idx] + 1)
end

function get_GammaPA(p0, p, pmax, e0, Gamma0, param::AbstractVector)
    p_interface = exp.(range(log(p0), log(pmax), length=length(param)+1))
    e_interface = zeros(length(p_interface))
    Gamma_interface = zeros(length(p_interface))

    e_interface[1] = e0
    Gamma_interface[1] = Gamma0

    for i in 2:length(p_interface)
        Gamma_interface[i] = Gamma_interface[i-1]*(p_interface[i]/p_interface[i-1])^(param[i-1])
        e_interface[i] = e_interface[i-1]
    end

    return (p0, p, pmax, Gamma0, param) -> GammaPA(p0, p, pmax, Gamma0, param, p_interface, e_interface, Gamma_interface)
end

function e(p0, p, pmax, e0, Gamma0, param::AbstractVector; GammaF=GammaCheb)
    return e0 + (p - p0) + quadgk(x -> GammaF(p0, x, pmax, Gamma0, param), p0, p, rtol=1e-12, atol=1e-12)[1] 
end

function cs2(p0, p, pmax, Gamma0, param::AbstractVector; GammaF=GammaCheb)
    return 1 / (1 + GammaF(p0, p, pmax, Gamma0, param))
end

function get_eos(param, crustfile; pmax=1.0e3, ncrust=100, npoints=500, logspacing=false, pa_eos=false)
    p_crust, e_crust, cs2_crust = load_crust(crustfile) 

    if !issorted(p_crust) && !issorted(e_crust)
        mask = get_monotonic_filter(p_crust, e_crust)
        p_crust = p_crust[mask]
        e_crust = e_crust[mask]
        cs2_crust = cs2_crust[mask]
    end

    e_crust_itp = DataInterpolations.PCHIPInterpolation(e_crust, p_crust; extrapolation=ExtrapolationType.Extension)  
    p_crust_itp = DataInterpolations.PCHIPInterpolation(p_crust, e_crust; extrapolation=ExtrapolationType.Extension)
    cs2_crust_itp = DataInterpolations.PCHIPInterpolation(cs2_crust, p_crust; extrapolation=ExtrapolationType.Extension)

    e0 = 150.0
    p0 = p_crust_itp(e0)
    cs20 = cs2_crust_itp(p0)

    Gamma0 = 1/cs20 - 1

    pout_crust = range(p_crust[1], p0, length=ncrust)
    eout_crust = e_crust_itp.(pout_crust)
    cs2out_crust = cs2_crust_itp.(pout_crust)

    δp = (pmax - p0)/ (npoints - 1)
    pout = range(p0+δp, pmax, length=npoints)
    if logspacing
        pout = exp.(range(log(p0 + δp), log(pmax), length=npoints))
    end

    GammaF = GammaCheb
    if pa_eos
        GammaF = get_GammaPA(p0, pout, pmax, e0, Gamma0, param)
    end

    eout = [e(p0, p, pmax, e0, Gamma0, param; GammaF=GammaF) for p in pout]
    cs2out = [cs2(p0, p, pmax, Gamma0, param; GammaF=GammaF) for p in pout]

    ptotal = vcat(collect(pout_crust), collect(pout))
    etotal = vcat(collect(eout_crust), collect(eout))
    cs2total = vcat(collect(cs2out_crust), collect(cs2out))

    return ptotal, etotal, cs2total
end