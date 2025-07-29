import argparse
import requests
from pyproj import Transformer

API_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_PARAMS = {
    "hourly": "temperature_2m,relativehumidity_2m,wind_speed_10m,wind_direction_10m",
    "forecast_days": 1,
}

def fetch_weather(lat: float, lon: float):
    params = dict(DEFAULT_PARAMS)
    params.update({"latitude": lat, "longitude": lon})
    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def wgs84_to_webmercator(lat: float, lon: float):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch weather data for GPS coordinates")
    parser.add_argument("lat", type=float, help="Latitude in degrees")
    parser.add_argument("lon", type=float, help="Longitude in degrees")
    args = parser.parse_args()

    data = fetch_weather(args.lat, args.lon)
    mercator = wgs84_to_webmercator(args.lat, args.lon)

    print(f"Coordinates (lat, lon): {args.lat}, {args.lon}")
    print(f"Web Mercator: {mercator[0]:.2f}, {mercator[1]:.2f}")
    print("Sample hourly data:")
    hourly = data.get("hourly", {})
    for key, values in hourly.items():
        if isinstance(values, list) and values:
            print(f"  {key}: {values[0]}")

if __name__ == "__main__":
    main()
