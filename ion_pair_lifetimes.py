#!/usr/bin/env python3
"""
ion_pair_lifetimes.py

Compute ion-pair lifetimes (continuous and intermittent) for Na-Cl pairs.

Usage:
  python ion_pair_lifetimes.py -t md_1ns_test_gk_unwrapped.xtc -p md_1ns_test_gk.tpr -n index.ndx -c 0.35

Outputs:
  - pair_acf.txt    : time(ps)  C(t)  (intermittent ACF)
  - pair_tau.txt    : scalar lifetime tau (s and ps)
  - survival.txt    : time(ps)  S_c(t) (continuous survival fraction)
  - pair_matrix.npz : boolean matrix h(t) saved for further analysis (optional)
"""
import argparse, numpy as np
import MDAnalysis as mda
from scipy import integrate

def read_ndx_groups(ndxfile):
    groups = {}
    name = None
    atoms = []
    with open(ndxfile) as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
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

def compute_h_matrix(u, na_idx, cl_idx, cutoff):
    # returns h: shape (nframes, n_na, n_cl) boolean (True if within cutoff)
    n_na = len(na_idx)
    n_cl = len(cl_idx)
    frames = []
    h_list = []
    for ts in u.trajectory:
        pos = u.atoms.positions  # nm
        na_pos = pos[na_idx]     # (n_na, 3)
        cl_pos = pos[cl_idx]     # (n_cl, 3)
        # compute pairwise distances
        # result shape (n_na, n_cl)
        d = np.linalg.norm(na_pos[:, None, :] - cl_pos[None, :, :], axis=2)
        h = (d <= cutoff)  # boolean
        h_list.append(h)
    h_arr = np.array(h_list, dtype=bool)  # (nframes, n_na, n_cl)
    times = np.array([ts.time for ts in u.trajectory])  # ps
    return times, h_arr

def intermittent_acf(times, h_arr):
    # flatten pairs: for ACF we treat each pair equally
    nframes = h_arr.shape[0]
    npairs = h_arr.shape[1]*h_arr.shape[2]
    # reshape to (nframes, npairs)
    H = h_arr.reshape(nframes, npairs).astype(float)
    mean_h = H.mean()
    if mean_h == 0:
        raise SystemExit("No pairs within cutoff in entire trajectory (mean_h==0).")
    # compute C(t) = <h(0)h(t)> / <h>
    # use FFT-based autocorrelation per pair then average
    # we compute numerator ACF(t) = (1/(N-t)) sum_i sum_start h_i(start)*h_i(start+t)
    N = nframes
    # compute for each pair
    ac_sum = np.zeros(N)
    for j in range(H.shape[1]):
        y = H[:, j] - 0.0  # no mean subtraction for numerator
        # FFT method:
        nx = N
        f = np.fft.rfft(y, n=2*nx)
        ac_full = np.fft.irfft(f * np.conjugate(f), n=2*nx)[:nx]
        # unbiased normalization: divide by (N - lag)
        norm = np.array([(nx - k) for k in range(nx)], dtype=float)
        ac_sum += ac_full / norm
    # average over pairs
    ac_avg = ac_sum / H.shape[1]
    C_t = ac_avg / mean_h  # because denominator is <h>
    return times, C_t

def continuous_survival(times, h_arr):
    # S_c(t): fraction of initially bound pairs that remain bound continuously up to t
    nframes, n_na, n_cl = h_arr.shape
    # initial bound pairs at t=0
    init = h_arr[0]  # (n_na,n_cl) boolean
    idx_pairs = np.argwhere(init)
    n_init = len(idx_pairs)
    if n_init == 0:
        return times, np.zeros_like(times)
    surv = np.zeros(nframes)
    for lag in range(nframes):
        cnt = 0
        for (i,j) in idx_pairs:
            # check h_arr[0..lag, i, j] all True
            if np.all(h_arr[:(lag+1), i, j]):
                cnt += 1
        surv[lag] = cnt / n_init
    return times, surv

def characteristic_tau(times, C_t):
    # integrate C(t) from t=0 to end using trapezoid
    time_s = (times - times[0]) * 1e-12
    # trapezoid
    integral = np.trapz(C_t, time_s)
    return integral  # seconds

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-t','--traj', required=True)
    p.add_argument('-p','--tpr', required=True)
    p.add_argument('-n','--ndx', required=True)
    p.add_argument('-c','--cutoff', type=float, required=True, help='cutoff in nm')
    p.add_argument('--na','--na_name', default='NA', help='name of Na group in ndx')
    p.add_argument('--cl','--cl_name', default='CL', help='name of Cl group in ndx')
    args = p.parse_args()

    u = mda.Universe(args.tpr, args.traj)
    groups = read_ndx_groups(args.ndx)
    if args.na not in groups or args.cl not in groups:
        print("Available groups in ndx:", list(groups.keys()))
        raise SystemExit("Index groups for ions not found.")
    na_idx = groups[args.na]
    cl_idx = groups[args.cl]

    print("Frames:", len(u.trajectory), "Na:", len(na_idx), "Cl:", len(cl_idx))
    times, h_arr = compute_h_matrix(u, na_idx, cl_idx, args.cutoff)
    np.savez('pair_matrix.npz', times=times, h=h_arr)  # optional save

    # intermittent ACF
    t, C_t = intermittent_acf(times, h_arr)
    np.savetxt('pair_acf.txt', np.column_stack([t, C_t]), header='time_ps C(t)', comments='')

    # compute tau (s)
    tau_s = characteristic_tau(times, C_t)
    tau_ps = tau_s * 1e12
    with open('pair_tau.txt','w') as f:
        f.write(f"tau_intermittent = {tau_s:.6e} s  ({tau_ps:.3f} ps)\n")
    print("tau_intermittent:", tau_s, "s (", tau_ps, "ps )")

    # continuous survival
    t_s, S_c = continuous_survival(times, h_arr)
    np.savetxt('survival.txt', np.column_stack([t_s, S_c]), header='time_ps S_continuous(t)', comments='')
    print("Wrote pair_acf.txt, pair_tau.txt, survival.txt, pair_matrix.npz")

if __name__ == '__main__':
    main()

