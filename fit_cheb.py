import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from eos_cheb import load_crust, get_eos # Import from our other file

def cost_function(param, p_true, e_true, pmax):
    """
    Cost function for the optimizer.
    Returns a vector of residuals, as required by least_squares.
    """
    try:
        # 1. Generate the EoS for the current 'param' vector
        p_model, e_model, _ = get_eos(param, "in/ska.table", pmax=pmax)
    except Exception as e:
        # If integration fails (bad params), return a huge error
        print(f"Warning: EoS generation failed: {e}")
        return np.full(len(p_true), 1e10) # Return a large residual vector

    # 2. Create interpolator for the "true" EoS
    e_true_itp = PchipInterpolator(p_true, e_true, extrapolate=True)

    # 3. Filter model EoS to match the core
    e0 = 150.0
    mask = e_model >= e0
    e_model_core = e_model[mask]
    p_model_core = p_model[mask]

    # 4. Get the "true" energy at the model's pressure points
    e_true_at_model_p = e_true_itp(p_model_core)

    # 5. Calculate and return the vector of residuals
    residuals = np.log(e_model_core / e_true_at_model_p)
    
    # Check for NaNs/Infs which can break the optimizer
    if not np.all(np.isfinite(residuals)):
         print("Warning: Non-finite residuals found.")
         return np.full(len(residuals), 1e10)

    return residuals

def fit_param(p_true, e_true, n_param=5, pmax=700.0):
    """
    Fits the EoS parameters using Scipy's least_squares.
    """
    
    # 1. Define the objective function (a "lambda" that holds extra args)
    objective = lambda p: cost_function(p, p_true, e_true, pmax)

    # 2. Set initial parameters (small, non-zero)
    initial_param = np.random.rand(n_param) * 0.1 - 0.05
    # initial_param = np.zeros(n_param) # Can also try 0.0

    # 3. Define the "box" constraints
    lower_bounds = np.full(n_param, -3.0)
    upper_bounds = np.full(n_param, 3.0)

    # 4. Run the optimization
    print(f"Starting optimization with {n_param} parameters...")
    result = least_squares(
        objective,
        initial_param,
        method='trf',  # Trust Region Reflective (good for bounds)
        bounds=(lower_bounds, upper_bounds),
        verbose=2,     # Print progress (0=silent, 1=summary, 2=full)
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12
    )
    
    return result

def test_fit_eos():
    """
    Main function to run the fit and plot results.
    Matches your 'test_fit_eos' script.
    """
    p_true, e_true, cs2_true = load_crust("in/ska.table")
    n_param = 5
    pmax = 700.0 # Match pmax in test script

    # 1. Fit parameters
    result = fit_param(p_true, e_true, n_param=n_param, pmax=pmax)
    best_params = result.x
    
    # 2. Calculate final chi-squared (using the returned residuals)
    residuals = result.fun
    N = len(residuals)
    final_chi2 = np.sum(residuals**2) / N

    print("\n--- Optimization Summary ---")
    print(f"Final Chi-squared error: {final_chi2}")
    print(f"Converged: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final parameters: {best_params}")
    print("----------------------------")

    # 3. Generate the final fitted EoS
    p_model, e_model, cs2_model = get_eos(best_params, "in/ska.table", pmax=pmax)

    # 4. Plot EoS comparison
    plt.figure(figsize=(12, 5))
    
    # Plot 1: e vs p
    plt.subplot(1, 2, 1)
    plt.loglog(e_true, p_true, label="True EOS (ska.table)", color="blue")
    plt.loglog(e_model, p_model, label=f"Fitted EOS (n={n_param})", color="red", linestyle="--")
    plt.axvline(150.0, color="black", linestyle=":", label="e0 = 150.0")
    plt.xlabel("Energy Density (e)")
    plt.ylabel("Pressure (p)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Plot 2: e vs cs2
    plt.subplot(1, 2, 2)
    plt.semilogx(e_true, cs2_true, label="True cs2", color="green")
    plt.semilogx(e_model, cs2_model, label="Fitted cs2", color="orange", linestyle="--")
    plt.axvline(150.0, color="black", linestyle=":", label="e0 = 150.0")
    plt.xlabel("Energy Density (e)")
    plt.ylabel("Sound Speed (cs2)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.suptitle(f"Chebyshev Fit Result ($\\chi^2$ = {final_chi2:.2e})")
    plt.tight_layout()
    plt.savefig("out/eos_fit_test_python.png")
    print("Saved plots to 'out/eos_fit_test_python.png'")

if __name__ == "__main__":
    # Ensure you have the 'out' directory
    import os
    if not os.path.exists("out"):
        os.makedirs("out")
    
    test_fit_eos()