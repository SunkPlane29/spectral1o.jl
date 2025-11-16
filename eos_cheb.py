import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad, solve_ivp
from numpy.polynomial import chebyshev

def load_crust(file):
    """
    Loads crust file.
    Assumes columns: ... p(3), cs2(4), e(5) ... (0-indexed)
    This matches the Julia code's 1-indexed columns 4, 5, 6.
    """
    data = np.loadtxt(file)
    p = data[:, 3]
    cs2 = data[:, 4]
    e = data[:, 5]
    return p, e, cs2

def GammaCheb(p0, p, pmax, Gamma0, param):
    """
    Causal Chebyshev-based Upsilon(p) function.
    [cite_start]This directly implements Eq. 3 from paper 2409.19421v1 [cite: 524]
    [cite_start]and matches your Julia GammaCheb function[cite: 13].
    """
    # Map pressure to the Chebyshev domain [-1, 1]
    y = -1.0 + 2.0 * np.log(p / p0) / np.log(pmax / p0)
    
    # Evaluate the Chebyshev series: Sum(param[i] * T_i(y))
    # np.polynomial.chebyshev.chebval is the direct equivalent of
    # your param_new/chepolsum logic in Julia.
    cheb_sum = chebyshev.chebval(y, param)
    
    # [cite_start]Apply the (1+y) factor and exponential form [cite: 524]
    return Gamma0 * np.exp(cheb_sum * (1 + y))

def e_func_integrand(x, p0, pmax, Gamma0, param, GammaF):
    """Helper function for integration, as quad needs a simple f(x)"""
    return GammaF(p0, x, pmax, Gamma0, param)

def e_func(p, p0, pmax, e0, Gamma0, param, GammaF):
    """
    Calculates energy density e(p) by integrating Upsilon(p).
    [cite_start]This implements Eq. A2 from paper 2409.19421v1[cite: 721],
    [cite_start]matching your Julia 'e' function [cite: 3-4].
    
    Note: This function is scalar (integrates one p at a time).
    """
    integral, _ = quad(
        e_func_integrand,
        p0,
        p,
        args=(p0, pmax, Gamma0, param, GammaF),
        epsabs=1e-12,
        epsrel=1e-12
    )
    # e(p) = e0 + (p-p0)/c^2 + (1/c^2) * integral(Upsilon(p')) dp'
    # Assuming c=1, as in your Julia code.
    return e0 + (p - p0) + integral

def cs2_func(p, p0, pmax, Gamma0, param, GammaF):
    """
    Calculates sound speed cs2(p) from Upsilon(p).
    [cite_start]Matches your Julia 'cs2' function[cite: 4].
    This function is vectorized.
    """
    Upsilon = GammaF(p0, p, pmax, Gamma0, param)
    # v^2 = c^2 / (1 + Upsilon)
    # Assuming c=1
    return 1.0 / (1.0 + Upsilon)

def get_eos(param, crustfile, pmax=1.0e3, ncrust=100, npoints=500, logspacing=False):
    """
    Generates a full EoS by stitching a crust to a Chebyshev core.
    This matches your Julia 'get_eos' function's logic.
    """
    
    # 1. Load and interpolate the crust
    p_crust, e_crust, cs2_crust = load_crust(crustfile)
    
    # Sort by pressure to ensure interpolators work
    sort_mask = np.argsort(p_crust)
    p_crust = p_crust[sort_mask]
    e_crust = e_crust[sort_mask]
    cs2_crust = cs2_crust[sort_mask]

    # Note: Scipy's PchipInterpolator(x, y) matches Julia's
    # DataInterpolations.PCHIPInterpolation(y, x)
    e_crust_itp = PchipInterpolator(p_crust, e_crust, extrapolate=True)
    p_crust_itp = PchipInterpolator(e_crust, p_crust, extrapolate=True)
    cs2_crust_itp = PchipInterpolator(p_crust, cs2_crust, extrapolate=True)

    # 2. Define the matching point
    e0 = 150.0
    p0 = p_crust_itp(e0)
    cs20 = cs2_crust_itp(p0)
    
    if cs20 <= 0 or cs20 > 1.0:
        print(f"WARNING: Acausal matching point! cs20 = {cs20}")
        
    Gamma0 = (1.0 / cs20) - 1.0
    
    # 3. Generate crust pressure points
    pout_crust = np.linspace(p_crust[0], p0, ncrust)
    eout_crust = e_crust_itp(pout_crust)
    cs2out_crust = cs2_crust_itp(pout_crust)

    # 4. Generate core pressure points
    if logspacing:
        pout_core = np.logspace(np.log10(p0), np.log10(pmax), npoints)
    else:
        pout_core = np.linspace(p0, pmax, npoints)
    
    # We must skip p0 to avoid duplicates
    pout_core = pout_core[1:]
    npoints = len(pout_core) # new length

    # 5. Calculate core EoS
    # Set the EoS function to use
    GammaF = GammaCheb

    # Define the ODE function d_e/d_p = 1 + Upsilon(p)
    # The 'e' argument is required by solve_ivp but not used here
    def ode_func(p, e, p0, pmax, Gamma0, param, GammaF):
        return 1.0 + GammaF(p0, p, pmax, Gamma0, param)

    try:
        # Solve the ODE from p0, evaluating at all core points
        sol = solve_ivp(
            ode_func,
            t_span=(p0, pout_core[-1]), # Solve from p0 to last p-point
            y0=[e0],                   # Initial condition: e(p0) = e0
            t_eval=pout_core,          # Points to get the solution at
            args=(p0, pmax, Gamma0, param, GammaF), # Extra args
            method='RK45',             # Standard, fast solver
            rtol=1e-9, atol=1e-9       # Good precision
        )
        
        if not sol.success:
            # If solver fails, raise an error to be caught by optimizer
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
            
        eout_core = sol.y[0]

    except Exception as e:
        # If GammaF(p) explodes with bad params, it can fail.
        # We must catch this for the optimizer.
        print(f"Warning: solve_ivp failed ({e}). Returning bad fit.")
        # Return an array of Infs to signal a failed step
        eout_core = np.full(npoints, np.inf)

    # Calculate cs2(p) (vectorized, already fast)
    cs2out_core = cs2_func(pout_core, p0, pmax, Gamma0, param, GammaF)
    
    # # Calculate e(p) (scalar, needs a loop)
    # eout_core = np.zeros(npoints)
    # for i in range(npoints):
    #     eout_core[i] = e_func(pout_core[i], p0, pmax, e0, Gamma0, param, GammaF)
        
    # # Calculate cs2(p) (vectorized)
    # cs2out_core = cs2_func(pout_core, p0, pmax, Gamma0, param, GammaF)

    # 6. Combine crust and core
    ptotal = np.concatenate((pout_crust, pout_core))
    etotal = np.concatenate((eout_crust, eout_core))
    cs2total = np.concatenate((cs2out_crust, cs2out_core))

    return ptotal, etotal, cs2total