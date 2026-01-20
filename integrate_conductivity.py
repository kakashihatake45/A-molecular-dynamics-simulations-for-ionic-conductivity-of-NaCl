#!/usr/bin/env python3

"""
This script calculates the Green-Kubo ionic conductivity from the output
of gmx analyze -ac.

It reads a multi-column .xvg file (e.g., from gmx analyze), sums the
ACF components (x, y, z), integrates the total ACF, and applies the
full Green-Kubo formula to get conductivity in SI units (S/m).
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import warnings

def calculate_conductivity(volume_nm3, acf_file="conductivity_ac.xvg", plot_file="conductivity_gk_plot.png"):
    """
    Calculates ionic conductivity from a GROMACS current autocorrelation file.

    Args:
        volume_nm3 (float): The average simulation box volume in nm^3.
        acf_file (str): The path to the ACF file (output of 'gmx analyze -ac').
        plot_file (str): The name of the output plot file.

    Returns:
        float: The calculated conductivity in S/m, or None if calculation fails.
    """

    # --- Constants for SI Unit Conversion ---
    T = 298.15  # K (Temperature of your simulation, update if different)
    KB = 1.380649e-23  # J/K (Boltzmann constant)
    E_CHARGE = 1.602176634e-19  # C (Elementary charge)
    PS_TO_S = 1e-12  # s/ps
    NM_TO_M = 1e-9   # m/nm

    # --- Load the ACF file ---
    try:
        # Load the file, skipping comments
        data = np.loadtxt(acf_file, comments=["#", "@"])
    except IOError:
        print(f"Error: Could not find '{acf_file}'.")
        print("Did you run 'gmx analyze -f current.xvg -ac conductivity_ac.xvg' first?")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred loading {acf_file}: {e}")
        sys.exit(1)

    # --- Validate data ---
    if data.ndim == 1 or data.shape[0] < 2:
        print(f"Error: The file '{acf_file}' has insufficient data (<= 1 row).")
        print("This can happen with very short simulations.")
        sys.exit(1)

    time_ps = data[:, 0]

    if data.shape[1] < 4:
        print(f"Error: The ACF file '{acf_file}' has fewer than 4 columns.")
        print("Expected format: time, ACF_x, ACF_y, ACF_z")
        sys.exit(1)

    # --- Process data ---
    # Sum the x, y, and z components of the ACF
    # GROMACS C(t) = <J(0)J(t)> in units of (e*nm/ps)^2
    acf_total = data[:, 1] + data[:, 2] + data[:, 3]

    # --- Integrate the total ACF vs. time ---
    # np.trapz(y, x) computes integral(y dx)
    # Units: (e*nm/ps)^2 * ps = e^2 * nm^2 / ps
    try:
        integral = np.trapz(acf_total, time_ps) # Units: [e^2 * nm^2 / ps]
    except Exception as e:
        print(f"An error occurred during integration: {e}")
        sys.exit(1)

    # --- Apply the final Green-Kubo formula ---
    # sigma = (1 / (3 * V_SI * KB * T)) * Integral_SI
    #
    # Integral_SI = integral * (E_CHARGE**2) * (NM_TO_M**2) / (PS_TO_S)
    # V_SI = volume_nm3 * (NM_TO_M**3)
    # KBT_SI = KB * T
    #
    # sigma = (1 / (3 * V_SI * KBT_SI)) * Integral_SI
    # sigma = (1 / (3 * volume_nm3 * NM_TO_M**3 * KBT_SI)) * (integral * E_CHARGE**2 * NM_TO_M**2 / PS_TO_S)
    # sigma = (integral * E_CHARGE**2) / (3 * volume_nm3 * KBT_SI * NM_TO_M * PS_TO_S)

    KBT_SI = KB * T
    sigma = (integral * E_CHARGE**2) / (3.0 * volume_nm3 * KBT_SI * NM_TO_M * PS_TO_S)

    print(f"\n--- Green-Kubo Conductivity Calculation ---")
    print(f"Average Box Volume (V):   {volume_nm3:.4f} nm^3")
    print(f"Temperature (T):          {T} K")
    print(f"K_B * T:                  {KBT_SI:.4e} J")
    print(f"Raw ACF Integral:         {integral:.6e} [e^2 * nm^2 / ps]")
    print(f"Final Conductivity (sigma): {sigma:.4f} S/m")

    # --- Plot the running integral (convergence) ---
    running_integral = np.zeros_like(time_ps)

    # Check for constant time step
    dt = time_ps[1] - time_ps[0]
    if np.allclose(np.diff(time_ps), dt):
        # Faster integration for constant dt
        running_integral = np.cumsum(acf_total) * dt
    else:
        # Use trapezoidal rule for running sum (slower, but more general)
        for i in range(1, len(time_ps)):
            running_integral[i] = np.trapz(acf_total[:i+1], time_ps[:i+1])

    running_sigma = (running_integral * E_CHARGE**2) / (3.0 * volume_nm3 * KBT_SI * NM_TO_M * PS_TO_S)

    # Use warnings.catch_warnings to suppress font warnings on some systems
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        plt.figure(figsize=(10, 6))
        plt.plot(time_ps, running_sigma, label="Integrated Conductivity (GK)")
        plt.xlabel("Integration Time (ps)")
        plt.ylabel("Conductivity (S/m)")
        plt.title("Green-Kubo Conductivity Convergence")
        plt.grid(True)
        plt.axhline(y=sigma, color='r', linestyle='--', label=f"Final Value: {sigma:.4f} S/m")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_file)
        print(f"\nPlot of integrated conductivity saved to '{plot_file}'")

    return sigma

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide the average box volume (in nm^3) as an argument.")
        print("Usage: python3 integrate_conductivity.py <volume_in_nm3> [acf_file]")
        sys.exit(1)

    try:
        v_nm3 = float(sys.argv[1])
    except ValueError:
        print(f"Error: Invalid volume '{sys.argv[1]}'. Please provide a number.")
        sys.exit(1)

    acf = "conductivity_ac.xvg"
    if len(sys.argv) > 2:
        acf = sys.argv[2]

    calculate_conductivity(v_nm3, acf_file=acf)
