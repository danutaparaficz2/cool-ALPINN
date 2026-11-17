import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import streamlit as st
import streamlit.components.v1 as components

worker_map = components.declare_component(
    "worker_map", path=str(Path(__file__).parent / "worker_map")
)

@st.cache_resource
def plotly_javascript():
    return get_plotlyjs()

st.set_page_config(page_title="COOL-ALPINN | Interactive Shop-Floor Simulator", layout="wide")
st.title("COOL-ALPINN: Interactive Shop-Floor Heat & Worker Safety Simulator")
st.markdown(
    "<div style='font-size: 1.25rem; font-weight: 600; line-height: 1.4; color: #0068c9;'>"
    "Drag the worker or a blue ventilation marker. "
    "Moving a vent recalculates the temperature field.</div>",
    unsafe_allow_html=True,
)
st.caption("Prototype only; not a safety assessment.")

# ==========================================
# SESSION STATE FOR DRAGGABLE WORKER (X, Y)
# ==========================================
if "worker_x" not in st.session_state:
    st.session_state.worker_x = 8.0
if "worker_y" not in st.session_state:
    st.session_state.worker_y = 6.0
if "pending_worker_position" in st.session_state:
    st.session_state.worker_x, st.session_state.worker_y = st.session_state.pop("pending_worker_position")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Shop-Floor Controls")
n_vents = st.sidebar.slider("Active Ventilation Systems", min_value=0, max_value=8, value=2, step=1)
airflow_multiplier = st.sidebar.slider(
    "Ventilation Airflow Speed", min_value=0.0, max_value=2.0, value=1.0, step=0.1,
    format="%.1fx", help="Relative airflow and cooling strength per active vent; 1.0x is the original speed."
)
n_sensors = 10

# ==========================================
# SHOP-FLOOR PHYSICS & PINN MODEL
# ==========================================
Lx, Ly = 20.0, 12.0
Nx, Ny = 80, 48
T_amb = 22.0
heat_load_threshold = 26.5
core_temp_alert = 38.0
alpha = 0.08
q_furnace = 4.2   # Calibrated so peak near furnace is ~58-65°C
q_press = 2.8     # Calibrated so peak near press is ~45-50°C
furnace_pos = (4.5, 6.0)
press_pos = (14.5, 7.5)

x_grid = np.linspace(0, Lx, Nx)
y_grid = np.linspace(0, Ly, Ny)
dx, dy = x_grid[1] - x_grid[0], y_grid[1] - y_grid[0]
X, Y = np.meshgrid(x_grid, y_grid)

if "vent_positions" not in st.session_state:
    st.session_state.vent_positions = [
        (6.5, 9.5), (11.0, 3.0), (15.5, 9.5), (8.5, 2.5),
        (3.0, 3.0), (4.0, 10.0), (17.0, 3.0), (12.0, 10.0)
    ]
active_vents = tuple(st.session_state.vent_positions[:n_vents])

def net_heat_source_np(x, y, vent_positions, airflow):
    furnace = q_furnace * np.exp(-((x - furnace_pos[0])**2 / (2 * 1.2**2) + (y - furnace_pos[1])**2 / (2 * 1.5**2)))
    press = q_press * np.exp(-((x - press_pos[0])**2 / (2 * 1.3**2) + (y - press_pos[1])**2 / (2 * 1.1**2)))
    cooling = np.zeros_like(x)
    for vx_pos, vy_pos in vent_positions:
        cooling += 1.1 * airflow * np.exp(-((x - vx_pos)**2 / (2 * 1.5**2) + (y - vy_pos)**2 / (2 * 1.5**2)))
    return furnace + press - cooling

def net_heat_source_torch(x, y, vent_positions, airflow):
    furnace = q_furnace * torch.exp(-((x - furnace_pos[0])**2 / (2 * 1.2**2) + (y - furnace_pos[1])**2 / (2 * 1.5**2)))
    press = q_press * torch.exp(-((x - press_pos[0])**2 / (2 * 1.3**2) + (y - press_pos[1])**2 / (2 * 1.1**2)))
    cooling = torch.zeros_like(x)
    for vx_pos, vy_pos in vent_positions:
        cooling = cooling + 1.1 * airflow * torch.exp(-((x - vx_pos)**2 / (2 * 1.5**2) + (y - vy_pos)**2 / (2 * 1.5**2)))
    return furnace + press - cooling

def ventilation_velocity(num_vents, airflow):
    return 0.025 * num_vents * airflow, 0.012 * num_vents * airflow

@st.cache_data
def simulate_ground_truth(vent_positions, airflow):
    vx_val, vy_val = ventilation_velocity(len(vent_positions), airflow)
    T = np.full((Ny, Nx), T_amb, dtype=np.float64)
    Q = net_heat_source_np(X, Y, vent_positions, airflow)
    original_dt = 0.2 * min(dx**2, dy**2) / alpha
    stable_dt = 0.9 / (2 * alpha * (1 / dx**2 + 1 / dy**2) + vx_val / dx + vy_val / dy)
    steps = int(np.ceil(5000 * original_dt / min(original_dt, stable_dt)))
    dt = 5000 * original_dt / steps
    for _ in range(steps):
        T_old = T.copy()
        d2x = (T_old[1:-1, 2:] - 2 * T_old[1:-1, 1:-1] + T_old[1:-1, :-2]) / dx**2
        d2y = (T_old[2:, 1:-1] - 2 * T_old[1:-1, 1:-1] + T_old[:-2, 1:-1]) / dy**2
        dx1 = (T_old[1:-1, 1:-1] - T_old[1:-1, :-2]) / dx
        dy1 = (T_old[1:-1, 1:-1] - T_old[:-2, 1:-1]) / dy
        T[1:-1, 1:-1] = T_old[1:-1, 1:-1] + dt * (alpha * (d2x + d2y) - (vx_val * dx1 + vy_val * dy1) + Q[1:-1, 1:-1])
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = T_amb
    return np.maximum(T, T_amb - 1.5)

class HeatPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 48), nn.Tanh(),
            nn.Linear(48, 48), nn.Tanh(),
            nn.Linear(48, 48), nn.Tanh(),
            nn.Linear(48, 1)
        )

    def forward(self, x, y):
        xn = 2.0 * (x / Lx) - 1.0
        yn = 2.0 * (y / Ly) - 1.0
        return T_amb + self.net(torch.cat([xn, yn], dim=1))

@st.cache_data
def solve_pinn_cached(vent_positions, num_sensors, airflow):
    T_true_local = simulate_ground_truth(vent_positions, airflow)
    rng = np.random.default_rng(7)
    sensor_xy = np.column_stack([rng.uniform(1.0, Lx - 1.0, num_sensors), rng.uniform(1.0, Ly - 1.0, num_sensors)])
    interp_true = RegularGridInterpolator((y_grid, x_grid), T_true_local)
    sensor_T = interp_true(sensor_xy[:, ::-1]) + rng.normal(0, 0.25, num_sensors)

    torch.manual_seed(42)
    model = HeatPINN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    xs = torch.tensor(sensor_xy[:, 0:1], dtype=torch.float32)
    ys = torch.tensor(sensor_xy[:, 1:2], dtype=torch.float32)
    Ts = torch.tensor(sensor_T[:, None], dtype=torch.float32)

    n_bc = 100
    x_bc = torch.cat([torch.linspace(0, Lx, n_bc), torch.linspace(0, Lx, n_bc),
                      torch.zeros(n_bc), torch.full((n_bc,), Lx)]).unsqueeze(1)
    y_bc = torch.cat([torch.zeros(n_bc), torch.full((n_bc,), Ly),
                      torch.linspace(0, Ly, n_bc), torch.linspace(0, Ly, n_bc)]).unsqueeze(1)
    T_bc = torch.full_like(x_bc, T_amb)

    xc = torch.rand(1200, 1) * Lx
    yc = torch.rand(1200, 1) * Ly
    vx_val, vy_val = ventilation_velocity(len(vent_positions), airflow)

    for _ in range(1500):
        opt.zero_grad()
        loss_data = torch.mean((model(xs, ys) - Ts) ** 2)
        loss_bc = torch.mean((model(x_bc, y_bc) - T_bc) ** 2)

        xc_req = xc.clone().requires_grad_(True)
        yc_req = yc.clone().requires_grad_(True)
        Tc = model(xc_req, yc_req)
        dTx, dTy = torch.autograd.grad(Tc, (xc_req, yc_req), torch.ones_like(Tc), create_graph=True)
        dTxx = torch.autograd.grad(dTx, xc_req, torch.ones_like(dTx), create_graph=True)[0]
        dTyy = torch.autograd.grad(dTy, yc_req, torch.ones_like(dTy), create_graph=True)[0]

        res = alpha * (dTxx + dTyy) - (vx_val * dTx + vy_val * dTy) + net_heat_source_torch(xc_req, yc_req, vent_positions, airflow)
        loss = 5.0 * loss_data + 2.0 * loss_bc + torch.mean(res ** 2)
        loss.backward()
        opt.step()

    Xf = torch.tensor(X.ravel()[:, None], dtype=torch.float32)
    Yf = torch.tensor(Y.ravel()[:, None], dtype=torch.float32)
    with torch.no_grad():
        Tp = model(Xf, Yf).numpy().reshape(Ny, Nx)
    return T_true_local, Tp, sensor_xy

T_true, T_pinn, sensor_xy = solve_pinn_cached(active_vents, n_sensors, airflow_multiplier)

# ==========================================
# WORKER CORE TEMPERATURE SIMULATION
# ==========================================
wx = st.session_state.worker_x
wy = st.session_state.worker_y
worker_dist = np.hypot(wx - furnace_pos[0], wy - furnace_pos[1])

t_mins = np.linspace(0, 60, 120)
path_x = np.clip(wx + 0.45 * np.sin(2 * np.pi * t_mins / 30.0), 0.5, Lx - 0.5)
path_y = np.clip(wy + 0.45 * np.cos(2 * np.pi * t_mins / 30.0), 0.5, Ly - 0.5)
worker_path = np.column_stack([path_x, path_y])

interp_true = RegularGridInterpolator((y_grid, x_grid), T_true)
interp_pinn = RegularGridInterpolator((y_grid, x_grid), T_pinn)
worker_temp = float(interp_pinn([[wy, wx]])[0])
worker_is_hot = worker_temp > heat_load_threshold
env_temp_true = interp_true(worker_path[:, ::-1])
env_temp_pinn = interp_pinn(worker_path[:, ::-1])

Tc_true = np.zeros_like(t_mins)
Tc_fused = np.zeros_like(t_mins)
Tc_wrist_only = np.zeros_like(t_mins)
Tc_true[0] = Tc_fused[0] = Tc_wrist_only[0] = 36.0

for i in range(1, len(t_mins)):
    dt_hr = (t_mins[i] - t_mins[i - 1]) / 60.0
    load_true = max(0.0, (env_temp_true[i] - heat_load_threshold) * 0.16) - 0.35 * (Tc_true[i - 1] - 37.0)
    Tc_true[i] = Tc_true[i - 1] + load_true * dt_hr

    load_fused = max(0.0, (env_temp_pinn[i] - heat_load_threshold) * 0.16) - 0.35 * (Tc_fused[i - 1] - 37.0)
    Tc_fused[i] = Tc_fused[i - 1] + load_fused * dt_hr

    wrist_proxy = T_amb + 0.32 * (env_temp_true[i] - T_amb)
    load_wrist = max(0.0, (wrist_proxy - heat_load_threshold) * 0.16) - 0.35 * (Tc_wrist_only[i - 1] - 37.0)
    Tc_wrist_only[i] = Tc_wrist_only[i - 1] + load_wrist * dt_hr

# ==========================================
# KPI BANNER
# ==========================================
peak_Tc = Tc_fused.max()
status_text = "HIGH RISK — Rest Break Needed" if peak_Tc >= core_temp_alert else ("MODERATE — Monitor Closely" if peak_Tc >= 37.6 else "SAFE — Normal Operation")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Worker Position", f"({wx:.1f}m, {wy:.1f}m)")
c2.metric("Local Workstation Temp", f"{env_temp_pinn.mean():.1f} °C")
c3.metric("Worker Core Temp (60m)", f"{peak_Tc:.2f} °C")
with c4:
    status_color = "#d32f2f" if peak_Tc >= core_temp_alert else "inherit"
    st.markdown(
        "<div style='font-size: 0.875rem; margin-bottom: 0.25rem;'>Safety Status</div>"
        f"<div style='font-size: 1.35rem; line-height: 1.35; overflow-wrap: anywhere; color: {status_color};'>{status_text}</div>",
        unsafe_allow_html=True,
    )

# ==========================================
# VISUALIZATION PANELS
# ==========================================
col_map, col_worker = st.columns([1.35, 1.0])

with col_map:
    t_min_full = float(np.floor(T_pinn.min()))
    t_max_full = float(np.ceil(T_pinn.max()))

    fig_map = go.Figure()

    # Continuous temperature heatmap covering full temperature range (no arrows)
    fig_map.add_trace(go.Heatmap(
        x=x_grid,
        y=y_grid,
        z=T_pinn,
        colorscale="Inferno",
        zmin=t_min_full,
        zmax=t_max_full,
        zsmooth="best",
        colorbar=dict(
            title="Temp (°C)",
            tickmode="linear",
            tick0=t_min_full,
            dtick=max(2, round((t_max_full - t_min_full) / 8))
        ),
        hovertemplate="X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Est. Temp: %{z:.1f} °C<extra></extra>"
    ))

    # Furnace and active vents
    fig_map.add_trace(go.Scatter(
        x=[furnace_pos[0]], y=[furnace_pos[1]],
        mode="markers+text", text=["Furnace"], textposition="bottom center",
        textfont=dict(color="red", size=11),
        marker=dict(symbol="square", size=16, color="red", line=dict(color="white", width=2)),
        name="Holding Furnace"
    ))
    if n_vents > 0:
        fig_map.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="circle", size=15, color="deepskyblue", line=dict(color="black", width=1.5)),
            name=f"Active Vents ({n_vents})"
        ))
    # Legend entry; the worker itself is drawn by the draggable overlay.
    fig_map.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(symbol="circle", size=12, color="white", line=dict(color="black", width=2.5)),
        name=f"Worker ({worker_dist:.1f}m from furnace)",
        hovertemplate=(
            f"Worker<br>X: {wx:.1f} m, Y: {wy:.1f} m"
            f"<br>Local temperature: {worker_temp:.1f} °C<extra></extra>"
        )
    ))

    fig_map.update_layout(
        title=dict(text="Shop-Floor Temperature", x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15)),
        xaxis=dict(title="Shop Floor X (meters)", range=[0, Lx], constrain="domain", fixedrange=True),
        yaxis=dict(title="Shop Floor Y (meters)", range=[0, Ly], scaleanchor="x", scaleratio=1, fixedrange=True),
        legend=dict(orientation="v", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        margin=dict(l=40, r=20, t=135, b=50),
        height=530,
        dragmode=False
    )

    event = worker_map(
        figure=json.loads(fig_map.to_json()),
        plotly_js=plotly_javascript(),
        worker_x=wx,
        worker_y=wy,
        is_hot=bool(worker_is_hot),
        is_alert=bool(peak_Tc >= core_temp_alert),
        worker_temp=worker_temp,
        vents=active_vents,
        furnace_position=furnace_pos,
        flame_distance=1.5,
        bounds=[0.8, Lx - 0.8, 0.8, Ly - 0.8],
        vent_bounds=[0.0, Lx, 0.0, Ly],
        key="floor_map",
    )

    if event and event["id"] != st.session_state.get("last_map_event"):
        st.session_state.last_map_event = event["id"]
        if event.get("kind") == "vent":
            index = event["index"]
            if not isinstance(index, int) or not 0 <= index < n_vents:
                raise ValueError(f"Invalid vent index: {index}")
            new_pos = (
                round(float(np.clip(event["x"], 0.0, Lx)), 1),
                round(float(np.clip(event["y"], 0.0, Ly)), 1),
            )
            if new_pos != st.session_state.vent_positions[index]:
                st.session_state.vent_positions[index] = new_pos
                st.rerun()
        elif event.get("kind") == "worker":
            new_x = float(np.clip(event["x"], 0.8, Lx - 0.8))
            new_y = float(np.clip(event["y"], 0.8, Ly - 0.8))
            new_x = round(0.8 + round((new_x - 0.8) / 0.2) * 0.2, 1)
            new_y = round(0.8 + round((new_y - 0.8) / 0.2) * 0.2, 1)
            if abs(new_x - st.session_state.worker_x) > 0.05 or abs(new_y - st.session_state.worker_y) > 0.05:
                st.session_state.pending_worker_position = (new_x, new_y)
                st.rerun()
        else:
            raise ValueError(f"Unknown map event: {event.get('kind')}")

with col_worker:
    fig_tc = go.Figure()
    fig_tc.add_trace(go.Scatter(x=t_mins, y=Tc_true, mode="lines", name="True Core Temp (Tc)",
                                line=dict(color="black", width=2.5)))
    fig_tc.add_trace(go.Scatter(x=t_mins, y=Tc_fused, mode="lines", name="COOL-ALPINN (PINN + Wearable)",
                                line=dict(color="red", width=2.5, dash="dash")))
    fig_tc.add_trace(go.Scatter(x=t_mins, y=Tc_wrist_only, mode="lines", name="Standalone Wristband (Fails)",
                                line=dict(color="royalblue", width=2.2, dash="dot")))

    fig_tc.add_hline(y=core_temp_alert, line_dash="dashdot", line_color="orange",
                     annotation_text=f"Heat Strain Alert ({core_temp_alert:.1f}°C)", annotation_position="top left")
    fig_tc.add_hrect(y0=core_temp_alert, y1=39.0, fillcolor="orange", opacity=0.12, line_width=0)

    fig_tc.update_layout(
        title=dict(text=f"Worker Core Temp at ({wx:.1f}m, {wy:.1f}m)", x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15)),
        xaxis=dict(title="Shift Duration (minutes)", range=[0, 60]),
        yaxis=dict(title="Core Body Temperature Tc (°C)", range=[35.8, 38.8]),
        legend=dict(orientation="v", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        margin=dict(l=40, r=20, t=135, b=50),
        height=530
    )
    st.plotly_chart(fig_tc, use_container_width=True, config={"displayModeBar": False})