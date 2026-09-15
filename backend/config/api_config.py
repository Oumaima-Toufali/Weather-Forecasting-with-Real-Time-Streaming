# Path: backend/config/api_config.py
"""Configuration pour Open-Meteo API"""

# Locations à surveiller (tu peux en ajouter d'autres)
LOCATIONS = {
    'casablanca': {
        'latitude': 33.5731,
        'longitude': -7.5898,
        'name': 'Casablanca'
    },
    'rabat': {
        'latitude': 34.0209,
        'longitude': -6.8416,
        'name': 'Rabat'
    },
    'marrakech': {
        'latitude': 31.6295,
        'longitude': -7.9811,
        'name': 'Marrakech'
    }
}

# Paramètres de polling
POLL_INTERVAL_SECONDS = 300  # 5 minutes (Open-Meteo permet polling fréquent)

# Variables météo à récupérer
WEATHER_VARIABLES = {
    'current': [
        'temperature_2m',
        'relative_humidity_2m',
        'precipitation',
        'wind_speed_10m',
        'wind_direction_10m',
        'cloud_cover',
        'pressure_msl'
    ],
    'hourly': [
        'temperature_2m',
        'relative_humidity_2m',
        'precipitation_probability',
        'precipitation',
        'wind_speed_10m',
        'cloud_cover'
    ]
}

FORECAST_DAYS = 2  # Nombre de jours de prévisions