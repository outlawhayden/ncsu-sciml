import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

# define constants
g = 1
l = 1
m = 1

# define initial condition
x0 = np.array([0.0, 1.0], np.float64)

def simulate_system(init_state, h, t_total):
    t_array = np.arange(0.0, t_total, h)
    n = len(t_array)
    hist = np.zeros((2,n), dtype = np.float64)

    state = init_state.reshape(2,)
    hist[:,0] = state
    for i,t in tqdm(enumerate(t_array)):
        state[0] = state[0] - np.sin(state[1])
        state[1] = state[1] + state[0]
        hist[:,i] = state
    return hist


print("generating data...")
pend_10_2000 = simulate_system(x0, 10, 2000)
np.savetxt("pend_10_2000.csv", pend_10_2000, delimiter=",")

pend_5_2000  = simulate_system(x0, 5, 2000)
np.savetxt("pend_5_2000.csv", pend_5_2000, delimiter=",")

pend_01_2000 = simulate_system(x0, 0.1, 2000)
np.savetxt("pend_01_2000.csv", pend_01_2000, delimiter = ",")
print("complete")
print("rendering out...")

plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams.update({'font.size': 18})

t_coords = np.arange(0.0, 2000, 10)
plt.plot(pend_10_2000[0,:], pend_10_2000[1,:])
plt.title("Pendulum Simulation (h = 10, t_max = 2000)")
plt.xlabel("p(t)")
plt.ylabel("q(t)")
plt.savefig("10_2000_fig.png", dpi=300, bbox_inches="tight")

mask = (t_coords >= 1800) & (t_coords <= 2000)

plt.figure(figsize = (12,8))
plt.plot(t_coords[mask], np.sin(pend_10_2000[1, mask]), label="sin(q(t))")
plt.plot(t_coords[mask], -np.cos(pend_10_2000[1, mask]), label="-cos(q(t))")
plt.xlabel("t")
plt.title("Pendulum Simulation Angles (h = 10, t_max = 2000)")
plt.legend(loc = "upper right")
plt.savefig("10_2000_fig_angles_zoom.png", dpi=300, bbox_inches="tight")


t_coords = np.arange(0.0, 2000, 5)
plt.figure(figsize = (8,8))
plt.plot(pend_5_2000[0,:], pend_5_2000[1,:])
plt.title("Pendulum Simulation (h = 5, t_max = 2000)")
plt.xlabel("p(t)")
plt.ylabel("q(t)")
plt.savefig("5_2000_fig.png", dpi=300, bbox_inches="tight")

mask = (t_coords >= 1800) & (t_coords <= 2000)

plt.figure(figsize = (12,8))
plt.plot(t_coords[mask], np.sin(pend_5_2000[1, mask]), label="sin(q(t))")
plt.plot(t_coords[mask], -np.cos(pend_5_2000[1, mask]), label="-cos(q(t))")
plt.xlabel("t")
plt.title("Pendulum Simulation Angles (h = 5, t_max = 2000)")
plt.legend(loc = "upper right")
plt.savefig("5_2000_fig_angles_zoom.png", dpi=300, bbox_inches="tight")



t_coords = np.arange(0.0, 2000, 0.1)
plt.figure(figsize = (8,8))
plt.plot(pend_01_2000[0,:], pend_01_2000[1,:])
plt.title("Pendulum Simulation (h = 0.1, t_max = 2000)")
plt.xlabel("p(t)")
plt.ylabel("q(t)")
plt.savefig("01_2000_fig.png", dpi=300, bbox_inches="tight")

mask = (t_coords >= 1800) & (t_coords <= 2000)

plt.figure(figsize = (12,8))
plt.plot(t_coords[mask], np.sin(pend_01_2000[1, mask]), label="sin(q(t))")
plt.plot(t_coords[mask], -np.cos(pend_01_2000[1, mask]), label="-cos(q(t))")
plt.xlabel("t")
plt.title("Pendulum Simulation Angles (h = 0.1, t_max = 2000)")
plt.legend(loc = "upper right")
plt.savefig("01_2000_fig_angles_zoom.png", dpi=300, bbox_inches="tight")
