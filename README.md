# AtmosphericLBM

This repository contains a minimal setup for an atmospheric simulation framework using the Lattice Boltzmann Method. The implementation currently provides stubs only for demonstration purposes.

For an overview of how GPS coordinates, open meteorological data, and an LBM solver could be combined into a full weather simulation system, see [docs/GPS_LBM_weather_guide.md](docs/GPS_LBM_weather_guide.md).

The `scripts/fetch_weather.py` utility demonstrates how to query the Open-Meteo API for real-world conditions given a pair of GPS coordinates:

```bash
python scripts/fetch_weather.py 40.0 -105.0
```

Install the optional Python dependencies with:

```bash
pip install -r requirements.txt
```
