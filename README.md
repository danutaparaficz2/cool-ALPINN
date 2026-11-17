# COOL-ALPINN

An interactive Streamlit prototype for exploring a simulated shop-floor temperature field and estimated worker core temperature.

## Run locally

Install the requirements and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run simulation.py
```

Drag the worker or fans on the map, or click the floor to move the worker. Changing ventilation settings or fan positions retrains the PINN and may take some time. `worker_map/index.html` must remain alongside `simulation.py` in the `worker_map` directory.

## Deploy

Connect this repository to [Streamlit Community Cloud](https://share.streamlit.io/), create an app, and select `simulation.py` as the entry point. The `requirements.txt` file supplies its Python dependencies.

This simulation is a prototype, not a validated safety or medical assessment.
