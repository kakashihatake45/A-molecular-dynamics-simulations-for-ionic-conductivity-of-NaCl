import numpy as np
import matplotlib.pyplot as plt

# Function to safely load numeric data from an .xvg file
def load_xvg(filename):
    times, vacf = [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # skip comments, headers, and blank lines
            if line.startswith(('#','@','&')) or len(line) == 0:
                continue
            try:
                t, v = map(float, line.split()[:2])
                times.append(t)
                vacf.append(v)
            except ValueError:
                # skip any line that cannot be converted to float
                continue
    return np.array(times), np.array(vacf)

# Load data
time, vacf = load_xvg('vacf_ION.xvg')

# Numerical integration (Green-Kubo)
D = (1/3) * np.trapz(vacf, x=time)  # units nm^2/ps
D_m2_s = D * 1e-18 / 1e-12         # convert nm^2/ps to m^2/s

print(f"Diffusion coefficient D = {D:.5f} nm^2/ps")
print(f"Diffusion coefficient D = {D_m2_s:.5e} m^2/s")

# Optional: plot
plt.plot(time, vacf)
plt.xlabel('Time [ps]')
plt.ylabel('VACF')
plt.title('Velocity Autocorrelation Function (Ion)')
plt.grid(True)
plt.show()

