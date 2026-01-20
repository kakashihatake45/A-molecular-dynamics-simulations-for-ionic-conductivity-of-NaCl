#!/usr/bin/env python3
import sys
import numpy as np

def load_xvg(fname):
    """Load a GROMACS .xvg file, ignoring metadata lines."""
    t = []
    vacf = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', '@', '&')):
                continue  # skip metadata or formatting lines
            try:
                cols = line.split()
                t.append(float(cols[0]))  # ps
                vacf.append(float(cols[1]))
            except ValueError:
                continue
    return np.array(t), np.array(vacf)

def compute_diffusion(vacf, time):
    """Compute diffusion coefficient using Green-Kubo relation."""
    # Integration of VACF over time: D = (1/3) ∫ <v(0)v(t)> dt
    D_nm2ps = (1/3) * np.trapz(vacf, time)  # nm^2/ps
    D_m2s = D_nm2ps * 1e-12  # convert to m^2/s
    return D_nm2ps, D_m2s

def compute_ionic_conductivity(D, N_ions, L_box, T=298):
    """Compute ionic conductivity using Nernst-Einstein relation."""
    # Constants
    q = 1.602176634e-19  # elementary charge in C
    kB = 1.380649e-23    # Boltzmann constant in J/K

    # Convert box length from nm to m
    L = L_box * 1e-9
    # Volume
    V = L**3
    # Ionic conductivity σ = (q^2 * N * D)/(kB * T * V)
    sigma = (q**2 * N_ions * D) / (kB * T * V)  # S/m
    return sigma

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 compute_diff_cond.py <vacf_file.xvg> <N_ions> <box_length_nm>")
        sys.exit(1)

    vacf_file = sys.argv[1]
    N_ions = int(sys.argv[2])
    L_box = float(sys.argv[3])

    time_ps, vacf = load_xvg(vacf_file)

    D_nm2ps, D_m2s = compute_diffusion(vacf, time_ps)
    sigma = compute_ionic_conductivity(D_m2s, N_ions, L_box)

    print(f"Diffusion coefficient: D = {D_nm2ps:.6e} nm^2/ps = {D_m2s:.6e} m^2/s")
    print(f"Ionic conductivity: σ = {sigma:.6e} S/m")

