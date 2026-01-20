#!/usr/bin/env python3
"""
compute_vacf_and_D.py

Compute VACF and Green-Kubo diffusion coefficient from trajectory.

Usage:
  python compute_vacf_and_D.py \
    -t md_1ns_test_gk.trr -p md_1ns_test_gk.tpr -n index.ndx \
    -g NA -o results_dir --dt 0.002

Notes:
 - dt: timestep in ps between frames (if MDAnalysis can't report ts.dt). Use e.g. 0.002 for 2 fs.
 - Group name (g) must exist in index.ndx (e.g. NA, CL, System).
 - Trajectory must contain velocities. Use .trr or a .xtc written with velocities.
"""
import argparse, os
import numpy as np
import MDAnalysis as mda
from scipy import integrate

def read_ndx_groups(ndxfile):
    groups = {}
    name = None
    atoms = []
    with open(ndxfile) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith('['):
                if name and atoms:
                    groups[name] = np.array(atoms, dtype=int)-1
                name = line.strip('[] ').strip()
                atoms = []
            else:
                atoms += [int(x) for x in line.split()]
        if name and atoms:
            groups[name] = np.array(atoms, dtype=int)-1
    return groups

def compute_vacf_fft(V): 
    # V: (nframes, natoms, 3) velocities in m/s
    # produce VACF(t) averaged over atoms and components: <v(0).v(t)>
    nframes, natoms, _ = V.shape
    # reshape to shape (nframes, natoms*3)
    Vflat = V.reshape(nframes, natoms*3)
    # subtract mean (usual zero mean not necessary but do it)
    Vflat -= Vflat.mean(axis=0, keepdims=True)
    nx = nframes
    # FFT-based autocorrelation for each flattened component, then sum
    f = np.fft.rfft(Vflat, n=2*nx, axis=0)
    acf_full = np.fft.irfft(f * np.conjugate(f), n=2*nx, axis=0)[:nx,:]  # shape (nx, natom*3)
    # normalize unbiased
    norm = np.array([(nx - k) for k in range(nx)], dtype=float)
    acf_unbiased = (acf_full.T / norm).T  # (nx, comps)
    # average over components
    acf = acf_unbiased.mean(axis=1)  # (nx,)
    return acf

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-t','--traj', required=True)
    p.add_argument('-p','--tpr', required=True)
    p.add_argument('-n','--ndx', required=True)
    p.add_argument('-g','--group', required=True, help='index group name (e.g. NA, CL, System)')
    p.add_argument('-o','--outdir', default='vacf_results')
    p.add_argument('--dt', type=float, default=None, help='time between frames in ps (if None uses ts.dt)')
    args = p.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    u = mda.Universe(args.tpr, args.traj)
    groups = read_ndx_groups(args.ndx)
    if args.group not in groups:
        print("Available groups:", list(groups.keys()))
        raise SystemExit("Group not in index.ndx")
    atom_inds = groups[args.group]
    natoms = len(atom_inds)
    print("Frames:", len(u.trajectory), "atoms in group:", natoms)

    # get dt from trajectory if possible
    if args.dt is None:
        try:
            dt = u.trajectory.dt  # ps
            if dt is None: raise Exception()
        except Exception:
            raise SystemExit("Frame timestep unknown: provide --dt in ps (e.g. 0.002 for 2 fs)")
    else:
        dt = args.dt

    # collect velocities: shape (nframes, natoms, 3), convert nm/ps -> m/s (1 nm/ps = 1000 m/s)
    conv = 1e3
    vel_list = []
    for ts in u.trajectory:
        v = u.atoms[atom_inds].velocities  # nm/ps if present
        if v is None or v.size==0:
            raise SystemExit("Trajectory does not contain velocities for atoms. Use a .trr with velocities.")
        vel_list.append(v.copy()*conv)
    V = np.array(vel_list)  # (nframes, natoms, 3)

    # compute VACF via FFT method
    acf = compute_vacf_fft(V)  # units (m/s)^2
    time = np.arange(len(acf))*dt  # ps

    # VACF to <v(0).v(t)> averaged over atoms & components
    # save vacf (time ps, acf m^2/s^2)
    np.savetxt(f"{args.outdir}/vacf.txt", np.column_stack([time, acf]), header='time_ps VACF_m2_per_s2', comments='')

    # integrate to get D(t): D = (1/3) * integral <v(0).v(t)> dt, time must be seconds
    time_s = time * 1e-12
    # cumulative integral (trapezoid)
    cumint = np.cumsum(0.5*(acf[:-1] + acf[1:]) * np.diff(time_s))
    D_t = (1.0/3.0) * cumint  # units m^2/s
    # write D vs time (use times time[1:])
    np.savetxt(f"{args.outdir}/D_vs_time.txt", np.column_stack([time[1:], D_t]), header='time_ps D_m2_per_s', comments='')
    # final D
    D_final = D_t[-1]
    with open(f"{args.outdir}/D_final.txt",'w') as f:
        f.write(f"D_final_m2_per_s {D_final:.6e}\n")
    print("Wrote", args.outdir, "/vacf.txt, D_vs_time.txt, D_final.txt")
    print("Final D (m^2/s) = {:.6e}".format(D_final))

if __name__ == '__main__':
    main()

