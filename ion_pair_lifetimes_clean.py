#!/usr/bin/env python3

import numpy as np
import sys

def calculate_lifetimes(filename, rc):
    """
    Calculates ion pair lifetimes from a GROMACS pairdist file.

    Parameters:
    filename (str): The input .xvg file from 'gmx pairdist'.
    rc (float): The cutoff distance (in nm) to define a bound pair.

    Returns:
    numpy.ndarray: An array of all bound-state lifetimes (in ps).
    """

    print(f"Loading data from {filename}...")
    try:
        # Load the .xvg file, skipping comment lines
        data = np.loadtxt(filename, comments=['@', '#'])
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    if data.ndim == 1:
        print("Error: Data file appears to have only one row (one frame).")
        return None

    time = data[:, 0]       # Time in ps
    dists = data[:, 1:]    # All pairwise distances (frames x pairs)
    dt = time[1] - time[0] # Time step in ps

    n_frames, n_pairs = dists.shape
    print(f"Found {n_frames} frames and {n_pairs} pairs.")
    print(f"Using cutoff rc = {rc:.3f} nm and timestep dt = {dt:.3f} ps")

    all_lifetimes = []

    # Iterate over each pair (each column)
    for j in range(n_pairs):
        # Get the distance time series for this one pair
        pair_dists = dists[:, j]
        
        # Create a boolean array: True if bound, False if unbound
        is_bound = pair_dists < rc

        i = 0
        while i < n_frames:
            # Look for the start of a bound event
            if is_bound[i]:
                start_frame = i
                # Follow the event until it's no longer bound
                while i < n_frames and is_bound[i]:
                    i += 1
                end_frame = i
                
                # Calculate the duration in frames and then in picoseconds
                duration_frames = end_frame - start_frame
                duration_ps = duration_frames * dt
                all_lifetimes.append(duration_ps)
            else:
                # Not bound, just move to the next frame
                i += 1

    return np.array(all_lifetimes)

# --- Main execution ---
if __name__ == "__main__":
    
    # --- 1. Get Input Filename ---
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        print("Error: Please provide an input file.")
        print("Usage: python3 ion_pair_lifetimes_clean.py <input_file.xvg>")
        sys.exit(1)
    
    # --- 2. Get Cutoff Distance (rc) ---
    try:
        rc_input = input("Enter cutoff distance rc (nm) [from RDF minimum, e.g. 0.35]: ")
        rc = float(rc_input)
    except ValueError:
        print("Invalid input. Please enter a number.")
        sys.exit(1)

    # --- 3. Calculate Lifetimes ---
    lifetimes = calculate_lifetimes(input_file, rc)

    # --- 4. Print Results ---
    if lifetimes is not None and len(lifetimes) > 0:
        print("\n--- Results ---")
        print(f"Total bound events found: {len(lifetimes)}")
        print(f"Mean lifetime:            {np.mean(lifetimes):.4f} ps")
        print(f"Median lifetime:          {np.median(lifetimes):.4f} ps")
        print(f"Min/Max lifetime:         {np.min(lifetimes):.4f} ps / {np.max(lifetimes):.4f} ps")

        # --- 5. Save Results ---
        out_life_file = input_file.replace('.xvg', '_lifetimes.txt')
        out_surv_file = input_file.replace('.xvg', '_survival.xvg')
        out_hist_file = input_file.replace('.xvg', '_histogram.xvg') # <-- New histogram file

        # Save the raw list of lifetimes
        np.savetxt(out_life_file, lifetimes, header='Lifetime (ps)', fmt='%.6f')
        print(f"\nSaved all {len(lifetimes)} lifetimes to: {out_life_file}")

        # Calculate and save the survival curve C(t)
        t_grid = np.linspace(0, np.max(lifetimes), 200)
        C_t = np.array([(lifetimes >= t).sum() / len(lifetimes) for t in t_grid])
        
        np.savetxt(out_surv_file, np.column_stack([t_grid, C_t]), 
                   header='t (ps)   C(t) - Survival Probability', fmt='%.6f')
        print(f"Saved survival curve to: {out_surv_file}")

        # --- 6. (NEW) SAVE HISTOGRAM ---
        # Create a histogram with 50 bins
        hist, bin_edges = np.histogram(lifetimes, bins=50)
        # Calculate the center of each bin for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        # Normalize the histogram (so the area sums to 1)
        hist_normalized = hist / (len(lifetimes) * (bin_edges[1] - bin_edges[0]))
        
        np.savetxt(out_hist_file, np.column_stack([bin_centers, hist_normalized]),
                   header='t (ps)   P(t) - Probability Density', fmt='%.6f')
        print(f"Saved normalized histogram to: {out_hist_file}")
        
    elif lifetimes is not None:
        print("\n--- Results ---")
        print("No bound events found with this cutoff distance.")
    else:
        print("Calculation failed.")
