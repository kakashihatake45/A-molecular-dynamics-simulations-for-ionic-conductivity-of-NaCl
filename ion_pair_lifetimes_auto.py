import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
import argparse

def find_ions(u):
    """Try to find Na and Cl atoms by residue, atom, or type names."""
    print("Detecting ion names automatically...")

    names_to_check = {}
    if hasattr(u.atoms, "resnames") and len(u.residues) > 0:
        names_to_check["resnames"] = set(str(r.resname).upper() for r in u.residues)
    if hasattr(u.atoms, "names"):
        names_to_check["names"] = set(str(a.name).upper() for a in u.atoms)
    if hasattr(u.atoms, "types"):
        names_to_check["types"] = set(str(a.type).upper() for a in u.atoms)

    print("Detected naming fields:", list(names_to_check.keys()))

    all_names = set()
    for k, v in names_to_check.items():
        all_names |= v

    # Try common variants of sodium and chloride
    possible_na = [n for n in all_names if "NA" in n or "SOD" in n]
    possible_cl = [n for n in all_names if "CL" in n or "CHL" in n]

    if not possible_na or not possible_cl:
        raise RuntimeError(f"Could not find Na/Cl in naming fields. Found: {all_names}")

    na_sel = " or ".join([f"name {n}" for n in possible_na])
    cl_sel = " or ".join([f"name {n}" for n in possible_cl])

    try:
        na_atoms = u.select_atoms(na_sel)
        cl_atoms = u.select_atoms(cl_sel)
    except Exception:
        # Try residue-level selection
        na_atoms = u.select_atoms(" or ".join([f"resname {n}" for n in possible_na if n]))
        cl_atoms = u.select_atoms(" or ".join([f"resname {n}" for n in possible_cl if n]))

    print(f"Found {len(na_atoms)} Na atoms and {len(cl_atoms)} Cl atoms.")
    return na_atoms, cl_atoms


def main():
    parser = argparse.ArgumentParser(description="Automatic Na–Cl ion pair detection and lifetime analysis")
    parser.add_argument("-t", "--trajectory", required=True)
    parser.add_argument("-p", "--topology", required=True)
    parser.add_argument("-c", "--cutoff", type=float, required=True)
    args = parser.parse_args()

    print(f"Loading trajectory: {args.trajectory}")
    u = mda.Universe(args.topology, args.trajectory)
    na_atoms, cl_atoms = find_ions(u)

    print(f"Using cutoff = {args.cutoff:.3f} nm")

    counts = []
    for ts in u.trajectory:
        d = distance_array(na_atoms.positions, cl_atoms.positions, box=ts.dimensions)
        pairs = np.sum(d < args.cutoff)
        counts.append(pairs)

    mean_pairs = np.mean(counts)
    print(f"Analyzed {len(u.trajectory)} frames.")
    print(f"Average Na–Cl pairs per frame = {mean_pairs:.2f}")

if __name__ == "__main__":
    main()


