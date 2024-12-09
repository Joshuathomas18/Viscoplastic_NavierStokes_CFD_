import numpy as np
import matplotlib.pyplot as plt

# Function to solve the Navier-Stokes equations for a specific configuration
def navier_stokes_solver(u_inlet, v_inlet, nx, ny, aspect_ratio, dt, steps):
    # Domain dimensions
    Lx = 2.0 if aspect_ratio == "2:1" else 1.0
    Ly = 1.0
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    nu = 0.01   # Kinematic viscosity
    rho = 1.0   # Density

    # Arrays
    u = np.zeros((ny, nx))  # x-velocity component
    v = np.zeros((ny, nx))  # y-velocity component
    p = np.zeros((ny, nx))  # Pressure field
    b = np.zeros((ny, nx))  # Source term for pressure Poisson equation

    # Boundary conditions
    u[:, 0] = u_inlet       # Left inlet velocity
    v[:, 0] = v_inlet       # Left inlet y-velocity

    # Pressure Poisson equation solver
    def pressure_poisson(p, b, dx, dy):
        pn = np.empty_like(p)
        for _ in range(100):  # Iterations for Poisson equation
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                              (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                             (2 * (dx**2 + dy**2)) -
                             dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        return p

    # Main solver loop
    for n in range(steps):
        un = u.copy()
        vn = v.copy()
        
        # Source term for Poisson equation
        b[1:-1, 1:-1] = (1 / dt * 
                         ((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx) +
                          (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)) -
                         ((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx))**2 -
                         2 * ((un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy) *
                              (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx)) -
                         ((vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy))**2)

        # Solve pressure Poisson equation
        p = pressure_poisson(p, b, dx, dy)
        
        # Update velocity fields
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        # Apply boundary conditions for velocity
        u[:, 0] = u_inlet    # Inlet
        u[:, -1] = 0         # Outlet
        u[0, :], u[-1, :] = 0, 0  # No-slip on top and bottom walls
        v[:, 0] = v_inlet    # Inlet
        v[:, -1] = 0         # Outlet
        v[0, :], v[-1, :] = 0, 0  # No-slip on top and bottom walls

    # Plotting the pressure contour and velocity vector field
    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, p, alpha=0.6, cmap="coolwarm")  # Pressure contours
    plt.colorbar(label="Pressure")
    plt.contour(X, Y, p, colors='black', linestyles='dotted')  # Pressure contours for clearer visualization
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])  # Velocity vectors (downsampled for clarity)
    plt.title(f"Pressure Contours and Velocity Vector at dt = {dt}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Return pressure and velocity profile data
    return u, v, p

# Run the simulation for a specific case
navier_stokes_solver(u_inlet=.1, v_inlet=1.18, nx=41, ny=41, aspect_ratio="1:1", dt=.01, steps=500)
