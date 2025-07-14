import matplotlib
matplotlib.use('TkAgg') # Or 'Qt5Agg', 'GTK3Agg' depending on your preference and installed libraries

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Define the N-pendulum dynamics ---
def pendulum_dynamics(t, Y, m_array, l_array, g):
    """
    Defines the differential equations for the N-pendulum.

    Args:
        t (float): Current time.
        Y (np.array): State vector [theta_1, ..., theta_N, dtheta_1, ..., dtheta_N].
        m_array (np.array): Array of masses [m_1, ..., m_N].
        l_array (np.array): Array of link lengths [l_1, ..., l_N].
        g (float): Acceleration due to gravity.

    Returns:
        np.array: Derivatives of the state vector [dtheta_1, ..., dtheta_N, ddtheta_1, ..., ddtheta_N].
    """
    N = len(m_array)
    thetas = Y[:N]
    dthetas = Y[N:]

    # Initialize matrices A and vector b (as per our interpretation of the equations)
    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(N):
        # Calculate elements for matrix A
        for j in range(N):
            sum_mk_A = 0
            for k_idx in range(max(i, j), N): # k starts from max(i, j) up to n-1 (0-indexed)
                sum_mk_A += m_array[k_idx]
            A[i, j] = sum_mk_A * l_array[j] * np.cos(thetas[i] - thetas[j])

        # Calculate elements for vector b
        sum_mk_g_sin_theta_i = 0
        for k_idx in range(i, N): # k starts from i up to n-1 (0-indexed)
            sum_mk_g_sin_theta_i += m_array[k_idx]
        b[i] = sum_mk_g_sin_theta_i * g * np.sin(thetas[i])

        for j in range(N):
            if i != j:
                sum_mk_sin_diff = 0
                for k_idx in range(max(i, j), N): # k starts from max(i, j) up to n-1 (0-indexed)
                    sum_mk_sin_diff += m_array[k_idx]
                b[i] += sum_mk_sin_diff * l_array[j] * dthetas[j]**2 * np.sin(thetas[i] - thetas[j])

    # Solve for angular accelerations: ddthetas = -A_inv @ b
    try:
        ddthetas = -np.linalg.solve(A, b) # More stable than direct inverse
    except np.linalg.LinAlgError:
        # Handle singular matrix case, e.g., by returning zeros or a small value
        print(f"Warning: Singular matrix encountered at t={t}. Returning zero accelerations.")
        ddthetas = np.zeros(N)


    return np.concatenate((dthetas, ddthetas))

# --- 2. Simulation parameters ---
N_pendulums = int(input("Enter the number of pendulums (N): ")) # Number of pendulum segments

# Masses (kg)
masses = np.array([1.0] * N_pendulums)
# Lengths (m)
lengths = np.array([0.5] * N_pendulums)
# Gravity (m/s^2)
gravity = 9.80665 # Standard gravity

# Initial conditions
# Initial angles (radians), measured from the vertical
initial_thetas = np.array([np.pi / (2 * (i + 1)) for i in range(N_pendulums)])
# Initial angular velocities (rad/s)
initial_dthetas = np.array([0.0] * N_pendulums)

# Ensure initial conditions match N_pendulums
if len(initial_thetas) != N_pendulums or len(initial_dthetas) != N_pendulums:
    raise ValueError("Initial conditions arrays must match N_pendulums.")

initial_state = np.concatenate((initial_thetas, initial_dthetas))

# Time span for simulation
t_span = (0, 60)  # Simulate from 0 to 20 seconds
t_eval = np.linspace(t_span[0], t_span[1], 3000) # Evaluate at 1000 points for smooth animation

# --- 3. Solve the ODEs ---
print("Simulating N-pendulum motion...")
sol = solve_ivp(
    fun=lambda t, Y: pendulum_dynamics(t, Y, masses, lengths, gravity),
    t_span=t_span,
    y0=initial_state,
    t_eval=t_eval,
    method='LSODA', # 'LSODA' is a good general-purpose method for stiff/non-stiff problems
    dense_output=True,
    rtol=1e-9,  # Relative tolerance for high precision
    atol=1e-10   # Absolute tolerance for high precision
)
print("Simulation complete.")

if not sol.success:
    print(f"Solver failed: {sol.message}")
    exit()

thetas_solution = sol.y[:N_pendulums, :] # Angular positions over time
dthetas_solution = sol.y[N_pendulums:, :] # Angular velocities over time

# --- 4. Calculate Cartesian coordinates for animation ---
def get_cartesian_coords(thetas, lengths):
    """
    Calculates the Cartesian coordinates of each mass.

    Args:
        thetas (np.array): Array of angular positions for a single time step.
        lengths (np.array): Array of link lengths.

    Returns:
        tuple: (x_coords, y_coords) of all masses, including the pivot (0,0).
    """
    N = len(thetas)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)

    for i in range(N):
        x[i+1] = x[i] + lengths[i] * np.sin(thetas[i])
        y[i+1] = y[i] - lengths[i] * np.cos(thetas[i]) # y-axis pointing down for pendulum visuals
    return x, y

# --- 5. Animate the pendulum ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-sum(lengths) * 1.1, sum(lengths) * 1.1)
ax.set_ylim(-sum(lengths) * 1.1, sum(lengths) * 1.1)
ax.set_title(f'{N_pendulums}-Pendulum Simulation')
ax.set_xlabel('X position (m)')
ax.set_ylabel('Y position (m)')
ax.grid(True)

line, = ax.plot([], [], 'o-', lw=2, markersize=8) # Pendulum arms and masses
trail_lines = [ax.plot([], [], '-', lw=0.5, alpha=0.7)[0] for _ in range(N_pendulums)] # Trail for each mass

def init():
    line.set_data([], [])
    for trail in trail_lines:
        trail.set_data([], [])
    return [line] + trail_lines

# Store trail history
trail_history = [[] for _ in range(N_pendulums)]
trail_length = 100 # Number of previous points to display for the trail

def update(frame):
    current_thetas = thetas_solution[:, frame]
    x_coords, y_coords = get_cartesian_coords(current_thetas, lengths)

    line.set_data(x_coords, y_coords)

    # Update trails
    for i in range(N_pendulums):
        trail_history[i].append((x_coords[i+1], y_coords[i+1]))
        if len(trail_history[i]) > trail_length:
            trail_history[i].pop(0) # Remove oldest point

        trail_x = [p[0] for p in trail_history[i]]
        trail_y = [p[1] for p in trail_history[i]]
        trail_lines[i].set_data(trail_x, trail_y)

    return [line] + trail_lines

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=t_eval[1]*1000 - t_eval[0]*1000) # Interval in ms

plt.show()

# Optional: Save the animation
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("N_pendulum_simulation.mp4", writer=writer)
