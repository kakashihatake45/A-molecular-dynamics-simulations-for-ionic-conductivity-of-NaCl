#!/usr/bin/env python3
import argparse

# constants
F = 96485.3329      # C/mol
R = 8.314462618     # J/(mol K)
T_default = 300.0   # K

def compute(D_slope_nm2_per_ps_na, D_slope_nm2_per_ps_cl, T=T_default, conc_M=1.0):
    # Input 'slope' should be d(MSD)/dt in nm^2/ps.
    slope_na = float(D_slope_nm2_per_ps_na)
    slope_cl = float(D_slope_nm2_per_ps_cl)

    # Diffusion: D = slope / 6 (3D)
    D_na_nm2_per_ps = slope_na / 6.0
    D_cl_nm2_per_ps = slope_cl / 6.0

    # Convert to m^2/s: 1 nm^2/ps = 1e-6 m^2/s
    D_na = D_na_nm2_per_ps * 1e-6
    D_cl = D_cl_nm2_per_ps * 1e-6

    # concentration in mol/m^3
    c = float(conc_M) * 1000.0

    sigma_NE = (F**2 / (R * T)) * ( (1**2) * D_na * c + (1**2) * D_cl * c )

    out = {
        "slope_na_nm2_per_ps": slope_na,
        "slope_cl_nm2_per_ps": slope_cl,
        "D_na_nm2_per_ps": D_na_nm2_per_ps,
        "D_cl_nm2_per_ps": D_cl_nm2_per_ps,
        "D_na_m2_s": D_na,
        "D_cl_m2_s": D_cl,
        "sigma_NE_S_m": sigma_NE
    }
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute D and sigma_NE from MSD slopes (d(MSD)/dt).")
    parser.add_argument("--slope-na", required=True, help="slope d(MSD)/dt for Na in nm^2/ps")
    parser.add_argument("--slope-cl", required=True, help="slope d(MSD)/dt for Cl in nm^2/ps")
    parser.add_argument("--T", type=float, default=T_default, help="Temperature in K")
    parser.add_argument("--conc", type=float, default=1.0, help="Concentration in M (mol/L)")
    args = parser.parse_args()

    res = compute(args.slope_na, args.slope_cl, T=args.T, conc_M=args.conc)

    print(f"Input slopes (d(MSD)/dt): Na = {res['slope_na_nm2_per_ps']:.6e} nm^2/ps, Cl = {res['slope_cl_nm2_per_ps']:.6e} nm^2/ps")
    print(f"D (nm^2/ps): Na = {res['D_na_nm2_per_ps']:.6e}, Cl = {res['D_cl_nm2_per_ps']:.6e}")
    print(f"D (m^2/s):  Na = {res['D_na_m2_s']:.6e}, Cl = {res['D_cl_m2_s']:.6e}")
    print(f"Nernst-Einstein conductivity (σ_NE) = {res['sigma_NE_S_m']:.6f} S/m (T={args.T} K, c={args.conc} M)")

