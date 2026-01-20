#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit

def fit_diffusion(msd_file, tmin=10, tmax=None):
    data = np.loadtxt(msd_file, comments=['#','@'])
    t = data[:,0]  # ps
    msd = data[:,1]  # nm^2
    if tmax is None:
        tmax = t[-1]/2
    mask = (t >= tmin) & (t <= tmax)
    slope, intercept = np.polyfit(t[mask], msd[mask], 1)
    # D = slope / (2d) ; for 3D D = slope / 6 ; units nm^2/ps => convert to m^2/s: 1 nm^2/ps = 1e-18 m^2 /1e-12 s = 1e-6 m^2/s
    D_nm2_per_ps = slope / 6.0
    D_m2_per_s = D_nm2_per_ps * 1e-6
    return t, msd, slope, D_nm2_per_ps, D_m2_per_s

for label, fname in [("Na", "msd_na.xvg"), ("Cl", "msd_cl.xvg")]:
    t, msd, slope, D_nm2_per_ps, D_m2_per_s = fit_diffusion(fname, tmin=5, tmax=None)
    print(f"{label}: slope(msd) = {slope:.6f} nm^2/ps, D = {D_nm2_per_ps:.6e} nm^2/ps = {D_m2_per_s:.6e} m^2/s")
    # plot
    plt.scatter(t, msd, label=f"{label} MSD")
plt.xlabel("time [ps]")
plt.ylabel("MSD [nm^2]")
plt.legend()
plt.tight_layout()
plt.savefig("msd_combined.png", dpi=300)
plt.show()

