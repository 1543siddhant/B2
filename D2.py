# weather_analysis.py  -- short, complete solution (uses your API key directly)
import requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, folium
from datetime import datetime

API_KEY = "bf0b45f63b62d43e5f99a5dc84389af6"   # per your request
BASE_CURR = "http://api.openweathermap.org/data/2.5/weather"
BASE_FORE = "http://api.openweathermap.org/data/2.5/forecast"  # 5-day / 3-hour

def fetch_current(city, units='metric'):
    r = requests.get(BASE_CURR, params={'q': city, 'appid': API_KEY, 'units': units}, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_forecast(city, units='metric'):
    r = requests.get(BASE_FORE, params={'q': city, 'appid': API_KEY, 'units': units}, timeout=10)
    r.raise_for_status()
    return r.json()

def record_from_current(j):
    if not j: return None
    return {
        'dt': pd.to_datetime(j.get('dt', None), unit='s'),
        'temp': j.get('main', {}).get('temp'),
        'humidity': j.get('main', {}).get('humidity'),
        'wind': j.get('wind', {}).get('speed'),
        'precip': j.get('rain', {}).get('1h', 0) if j.get('rain') else 0
    }

def df_from_forecast(j):
    rows = []
    for item in j.get('list', []):
        dt = pd.to_datetime(item.get('dt', None), unit='s')
        temp = item.get('main', {}).get('temp')
        hum  = item.get('main', {}).get('humidity')
        wind = item.get('wind', {}).get('speed')
        # precipitation in forecast often under 'rain' with key '3h'
        precip = item.get('rain', {}).get('3h', 0) if item.get('rain') else 0
        rows.append({'dt': dt, 'temp': temp, 'humidity': hum, 'wind': wind, 'precip': precip})
    return pd.DataFrame(rows)

def analyze_and_plot(city):
    # fetch data
    curr = fetch_current(city)
    fore = fetch_forecast(city)
    df_fore = df_from_forecast(fore)
    row_curr = record_from_current(curr)
    if row_curr is not None:
        # add current as a single-row dataframe (avoid duplicates if same timestamp)
        df = pd.concat([pd.DataFrame([row_curr]), df_fore], ignore_index=True).drop_duplicates(subset=['dt']).sort_values('dt')
    else:
        df = df_fore.copy()

    df = df.dropna(subset=['temp']).reset_index(drop=True)
    df['date'] = df['dt'].dt.date

    # Daily aggregation
    daily = df.groupby('date').agg(
        mean_temp = ('temp','mean'),
        max_temp  = ('temp','max'),
        min_temp  = ('temp','min'),
        mean_hum  = ('humidity','mean'),
        sum_precip= ('precip','sum')
    ).reset_index()

    print(f"\nSummary for {city}:")
    print("Collected points:", len(df))
    print(daily)

    # Time series plot: temp, humidity, wind
    plt.figure(figsize=(10,5))
    plt.plot(df['dt'], df['temp'], marker='o', label='Temp (°C)')
    plt.plot(df['dt'], df['humidity'], marker='s', label='Humidity (%)')
    plt.plot(df['dt'], df['wind'], marker='^', label='Wind (m/s)')
    plt.title(f"Weather time-series: {city}")
    plt.xlabel("Datetime"); plt.xticks(rotation=30)
    plt.legend(); plt.tight_layout(); plt.show()

    # Daily precipitation bar plot
    plt.figure(figsize=(8,4))
    plt.bar(daily['date'].astype(str), daily['sum_precip'], color='skyblue')
    plt.title(f"Daily Precipitation (mm) - {city}"); plt.xlabel("Date"); plt.ylabel("Precipitation (mm)")
    plt.xticks(rotation=30); plt.tight_layout(); plt.show()

    # Correlation heatmap among numeric attributes (use aggregated daily)
    corr = daily[['mean_temp','mean_hum','max_temp','min_temp','sum_precip']].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu', center=0)
    plt.title(f"Correlation (daily) - {city}"); plt.tight_layout(); plt.show()

    # Geospatial map (city coords from current response)
    coord = curr.get('coord', {}) if curr else {}
    lat, lon = coord.get('lat'), coord.get('lon')
    if lat is not None and lon is not None:
        m = folium.Map(location=[lat, lon], zoom_start=6)
        popup = f"{city}<br>Last sample: {df['dt'].max()}<br>Temp: {df['temp'].iloc[-1]} °C"
        folium.Marker([lat, lon], popup=popup).add_to(m)
        m.save('weather_map.html')
        print("Saved map to weather_map.html")

# ---- run for a city (change CITY to any city name) ----
if __name__ == "__main__":
    CITY = "London"
    analyze_and_plot(CITY)
