import numpy as np

# Load RDF data
data = np.loadtxt("rdf_Na_Cl.xvg", comments=['@','#'])
r = data[:,0]   # distance in nm
g = data[:,1]   # RDF g(r)

# Parameters
N_Cl = 39                   # number of Cl atoms
V_box = 6.33**3             # box volume in nm^3, replace with your box length^3
rho_Cl = N_Cl / V_box       # number density of Cl in nm^-3

# Find first minimum
# Simple approach: look for first g(r) minimum after the first peak
peak_index = np.argmax(g)
# First minimum after peak
min_index = peak_index + np.argmin(g[peak_index:])

r_min = r[min_index]
print(f"First minimum at r = {r_min:.3f} nm")

# Integrate g(r) up to first minimum
dr = r[1] - r[0]  # assume uniform spacing
coord_number = 4*np.pi*rho_Cl*np.sum(g[:min_index+1] * r[:min_index+1]**2 * dr)

print(f"Average number of Cl ions around a Na in first shell: {coord_number:.2f}")

