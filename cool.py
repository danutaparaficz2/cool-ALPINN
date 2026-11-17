import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on device: {device}")

# ==========================================
# 1. SHOP FLOOR PARAMETERS & PHYSICS
# ==========================================
Lx, Ly = 20.0, 12.0       # Shop floor dimensions in meters (20m x 12m)
Nx, Ny = 100, 60          # Grid resolution for ground-truth FDM solver
T_ambient = 22.0          # Ambient / wall temperature (°C)
alpha = 0.08              # Effective thermal diffusivity (m^2/s)
vx, vy = 0.03, 0.01       # Mild ventilation air drift (m/s)

x_grid = np.linspace(0, Lx, Nx)
y_grid = np.linspace(0, Ly, Ny)
dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]
X, Y = np.meshgrid(x_grid, y_grid)  # Shape: (Ny, Nx)

def heat_source_np(x, y):
    """Simulates heat injection (°C/s) from two industrial machines."""
    # Source 1: Holding furnace at (4.5m, 6.0m)
    furnace = 18.0 * np.exp(-((x - 4.5)**2 / (2 * 1.1**2) + (y - 6.0)**2 / (2 * 1.4**2)))
    # Source 2: Extrusion press at (13.5m, 7.5m)
    press = 11.0 * np.exp(-((x - 13.5)**2 / (2 * 1.3**2) + (y - 7.5)**2 / (2 * 1.0**2)))
    return furnace + press

def heat_source_torch(x, y):
    """Differentiable heat source term for the PINN PDE residual."""
    furnace = 18.0 * torch.exp(-((x - 4.5)**2 / (2 * 1.1**2) + (y - 6.0)**2 / (2 * 1.4**2)))
    press = 11.0 * torch.exp(-((x - 13.5)**2 / (2 * 1.3**2) + (y - 7.5)**2 / (2 * 1.0**2)))
    return furnace + press

# ==========================================
# 2. GENERATE GROUND TRUTH (FINITE DIFFERENCE)
# ==========================================
def solve_ground_truth(iterations=8000):
    T = np.full((Ny, Nx), T_ambient, dtype=np.float64)
    Q = heat_source_np(X, Y)
    dt = 0.2 * min(dx**2, dy**2) / alpha

    for _ in range(iterations):
        T_old = T.copy()
        d2T_dx2 = (T_old[1:-1, 2:] - 2 * T_old[1:-1, 1:-1] + T_old[1:-1, :-2]) / dx**2
        d2T_dy2 = (T_old[2:, 1:-1] - 2 * T_old[1:-1, 1:-1] + T_old[:-2, 1:-1]) / dy**2
        # Upwind advection for ventilation drift
        dT_dx = (T_old[1:-1, 1:-1] - T_old[1:-1, :-2]) / dx
        dT_dy = (T_old[1:-1, 1:-1] - T_old[:-2, 1:-1]) / dy

        T[1:-1, 1:-1] = T_old[1:-1, 1:-1] + dt * (
            alpha * (d2T_dx2 + d2T_dy2) - (vx * dT_dx + vy * dT_dy) + Q[1:-1, 1:-1]
        )
        # Fixed ambient temperature along outer walls
        T[0, :] = T_ambient
        T[-1, :] = T_ambient
        T[:, 0] = T_ambient
        T[:, -1] = T_ambient

    return T

print("Simulating ground-truth shop floor microclimate...")
T_true = solve_ground_truth()

# ==========================================
# 3. PLACE SPARSE NOISY SENSORS
# ==========================================
# 12 sensor positions scattered across pillars/walls (not directly inside the furnace)
sensor_coords = np.array([
    [2.0, 2.5], [2.5, 9.5], [5.5, 3.0], [6.5, 9.0],
    [8.0, 6.0], [9.5, 2.0], [10.0, 10.0], [11.5, 5.5],
    [15.0, 3.5], [16.0, 9.5], [17.5, 6.0], [7.0, 6.0]
])

# Interpolate ground-truth temperatures at sensor locations & add realistic sensor noise
from scipy.interpolate import RegularGridInterpolator
interp_fn = RegularGridInterpolator((y_grid, x_grid), T_true)
sensor_temps_clean = interp_fn(sensor_coords[:, ::-1])
sensor_noise = np.random.normal(0, 0.35, size=sensor_temps_clean.shape)  # ±0.35°C noise
sensor_temps_noisy = sensor_temps_clean + sensor_noise

# ==========================================
# 4. DEFINE THE PINN ARCHITECTURE
# ==========================================
class HeatFlowPINN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        # Xavier initialization for smooth derivatives
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        # Normalize inputs to [-1, 1] for stable neural network gradients
        x_norm = 2.0 * (x / Lx) - 1.0
        y_norm = 2.0 * (y / Ly) - 1.0
        xy = torch.cat([x_norm, y_norm], dim=1)
        # Predict temperature offset above ambient
        return T_ambient + self.net(xy)

def compute_pde_residual(model, x, y):
    """Computes steady-state advection-diffusion PDE residual using autograd."""
    x.requires_grad_(True)
    y.requires_grad_(True)
    T = model(x, y)

    dT_dx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    dT_dy = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]

    d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dy2 = torch.autograd.grad(dT_dy, y, grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0]

    Q = heat_source_torch(x, y)
    # PDE: alpha * (T_xx + T_yy) - (vx * T_x + vy * T_y) + Q = 0
    residual = alpha * (d2T_dx2 + d2T_dy2) - (vx * dT_dx + vy * dT_dy) + Q
    return residual, dT_dx, dT_dy

# ==========================================
# 5. PREPARE TRAINING TENSORS & TRAIN PINN
# ==========================================
# Sensor tensors
x_sens = torch.tensor(sensor_coords[:, 0:1], dtype=torch.float32, device=device)
y_sens = torch.tensor(sensor_coords[:, 1:2], dtype=torch.float32, device=device)
T_sens = torch.tensor(sensor_temps_noisy[:, None], dtype=torch.float32, device=device)

# Boundary condition points (4 walls)
n_bc = 200
x_bc = torch.cat([
    torch.linspace(0, Lx, n_bc), torch.linspace(0, Lx, n_bc),
    torch.zeros(n_bc), torch.full((n_bc,), Lx)
]).unsqueeze(1).to(device)
y_bc = torch.cat([
    torch.zeros(n_bc), torch.full((n_bc,), Ly),
    torch.linspace(0, Ly, n_bc), torch.linspace(0, Ly, n_bc)
]).unsqueeze(1).to(device)
T_bc = torch.full_like(x_bc, T_ambient)

# Collocation points across entire 2D floor (enforces physics everywhere)
n_colloc = 2500
x_colloc = (torch.rand(n_colloc, 1, device=device) * Lx)
y_colloc = (torch.rand(n_colloc, 1, device=device) * Ly)

model = HeatFlowPINN(hidden_dim=64, num_layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)

print("Training Physics-Informed Neural Network...")
epochs = 5000
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # 1. Sensor data loss
    T_pred_sens = model(x_sens, y_sens)
    loss_data = torch.mean((T_pred_sens - T_sens) ** 2)

    # 2. Boundary wall loss
    T_pred_bc = model(x_bc, y_bc)
    loss_bc = torch.mean((T_pred_bc - T_bc) ** 2)

    # 3. Physics PDE residual loss
    res_pde, _, _ = compute_pde_residual(model, x_colloc, y_colloc)
    loss_pde = torch.mean(res_pde ** 2)

    # Weighted total loss
    loss = 5.0 * loss_data + 2.0 * loss_bc + 1.0 * loss_pde
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.4f} "
              f"(Sensor: {loss_data.item():.4f}, BC: {loss_bc.item():.4f}, PDE: {loss_pde.item():.4f})")

# ==========================================
# 6. EVALUATE CONTINUOUS FIELD & HEAT FLUX
# ==========================================
X_flat = torch.tensor(X.ravel()[:, None], dtype=torch.float32, device=device)
Y_flat = torch.tensor(Y.ravel()[:, None], dtype=torch.float32, device=device)

_, dT_dx_pred, dT_dy_pred = compute_pde_residual(model, X_flat, Y_flat)
with torch.no_grad():
    T_pred = model(X_flat, Y_flat).cpu().numpy().reshape(Ny, Nx)

# Fourier's law of heat conduction: q = -k * grad(T)
qx_pred = -dT_dx_pred.detach().cpu().numpy().reshape(Ny, Nx)
qy_pred = -dT_dy_pred.detach().cpu().numpy().reshape(Ny, Nx)

mae = np.mean(np.abs(T_true - T_pred))
print(f"\nReconstruction Mean Absolute Error across full 2D floor: {mae:.2f} °C")

# ==========================================
# 7. VISUALIZE RESULTS
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
vmin, vmax = T_ambient, max(T_true.max(), T_pred.max())

# Plot 1: Ground-Truth Temperature + Sensor Locations
im0 = axes[0].contourf(X, Y, T_true, levels=35, cmap="inferno", vmin=vmin, vmax=vmax)
axes[0].scatter(sensor_coords[:, 0], sensor_coords[:, 1], c="cyan", edgecolors="black", s=65, label="Sparse Sensors (n=12)")
axes[0].set_title("Ground-Truth Shop Floor Temp (°C)")
axes[0].set_xlabel("X (meters)")
axes[0].set_ylabel("Y (meters)")
axes[0].legend(loc="upper right")
plt.colorbar(im0, ax=axes[0])

# Plot 2: PINN Continuous Reconstruction + Heat Flow Vectors
im1 = axes[1].contourf(X, Y, T_pred, levels=35, cmap="inferno", vmin=vmin, vmax=vmax)
step = 5
axes[1].quiver(
    X[::step, ::step], Y[::step, ::step],
    qx_pred[::step, ::step], qy_pred[::step, ::step],
    color="white", alpha=0.85, scale=45
)
axes[1].set_title("PINN Reconstruction + Heat Flux (-∇T)")
axes[1].set_xlabel("X (meters)")
plt.colorbar(im1, ax=axes[1])

# Plot 3: Absolute Reconstruction Error
err = np.abs(T_true - T_pred)
im2 = axes[2].contourf(X, Y, err, levels=25, cmap="viridis")
axes[2].scatter(sensor_coords[:, 0], sensor_coords[:, 1], c="white", edgecolors="black", s=45)
axes[2].set_title(f"Absolute Error |T_true - T_PINN| (MAE: {mae:.2f}°C)")
axes[2].set_xlabel("X (meters)")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()