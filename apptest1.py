# -*- coding: utf-8 -*-
import streamlit as st
import googlemaps
import requests
from datetime import datetime
import pytz
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import osmnx as ox
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------------------
# Config
# -------------------------------
GOOGLE_API_KEY = "AIzaSyAX3v9OSj4Fg3Ad649BIRR13B09CidYNqc"
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

st.set_page_config(page_title="Smart Jogging Routes", page_icon="🏃‍♂️", layout="wide")

# -------------------------------
# Session state
# -------------------------------
if 'results_generated' not in st.session_state:
    st.session_state.results_generated = False
if 'route_data' not in st.session_state:
    st.session_state.route_data = None
if 'map_data' not in st.session_state:
    st.session_state.map_data = None

# -------------------------------
# Utility Functions
# -------------------------------
@st.cache_data(ttl=300)
def geocode_address(address):
    try:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            return geocode_result[0]['geometry']['location']
    except Exception as e:
        st.error(f"Geocoding error: {e}")
    return None

@st.cache_data(ttl=3600)
def get_timezone(lat, lng):
    try:
        timestamp = int(datetime.utcnow().timestamp())
        timezone_info = gmaps.timezone((lat, lng), timestamp)
        return timezone_info['timeZoneId']
    except Exception as e:
        st.error(f"Timezone error: {e}")
        return 'UTC'

@st.cache_data(ttl=3600)
def get_elevation(lat, lng):
    try:
        result = gmaps.elevation((lat, lng))
        return result[0]['elevation'] if result else 0
    except Exception as e:
        st.error(f"Elevation error: {e}")
        return 0

@st.cache_data(ttl=300)
def get_air_quality_data(lat, lng):
    url = f"https://airquality.googleapis.com/v1/currentConditions:lookup?key={GOOGLE_API_KEY}"
    payload = {"location": {"latitude": lat, "longitude": lng}, "languageCode": "en"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "currentConditions" in data:
                aqi_info = data["currentConditions"]["indexes"][0]
                return {
                    "aqi": aqi_info.get("aqi"),
                    "category": aqi_info.get("category"),
                    "dominant_pollutant": aqi_info.get("dominantPollutant", "unknown"),
                    "display_name": aqi_info.get("displayName", "Universal AQI")
                }
    except Exception as e:
        st.warning(f"AQI API failed: {e}")
    return {"aqi": 85, "category": "Moderate", "dominant_pollutant": "PM2.5", "display_name": "Fallback AQI"}

# -------------------------------
# Route Scoring with osmnx
# -------------------------------
def score_jogging_paths(place_name="Riga, Latvia"):
    G = ox.graph_from_place(place_name, network_type='walk')
    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()  # expose u,v,key

    if 'surface' not in edges.columns:
        edges['surface'] = None

    jogging_edges = edges[edges['highway'].isin(['footway', 'path', 'cycleway'])].copy()
    jogging_edges['length'] = jogging_edges['length'].fillna(0)
    jogging_edges['surface'] = jogging_edges['surface'].fillna('unknown')
    jogging_edges['surface_encoded'] = pd.factorize(jogging_edges['surface'])[0]

    jogging_edges['score'] = (
        0.5 * jogging_edges['length'] +
        10 * (jogging_edges['surface'] == 'paved').astype(int)
    )

    X = jogging_edges[['length', 'surface_encoded']]
    y = jogging_edges['score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    jogging_edges['predicted_score'] = model.predict(X)
    return G, jogging_edges

def generate_routes(G, jogging_edges, start_lat, start_lng, num_routes=3):
    routes = []
    start_node = ox.distance.nearest_nodes(G, start_lng, start_lat)
    top_edges = jogging_edges.sort_values("predicted_score", ascending=False).head(num_routes)

    for i, edge in top_edges.iterrows():
        u, v = edge["u"], edge["v"]
        try:
            path = nx.shortest_path(G, start_node, v, weight="length")
            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
            length_m = sum(ox.utils_graph.get_route_edge_attributes(G, path, "length"))
            routes.append({
                "coordinates": coords,
                "distance": length_m/1000,
                "avg_aqi": 80 + i*5,  # placeholder AQI
                "score": edge["predicted_score"]
            })
        except Exception:
            continue
    return routes

# -------------------------------
# Map Creation
# -------------------------------
def create_plotly_map(center_coords, aqi_data, routes):
    lat, lng = center_coords['lat'], center_coords['lng']
    fig = go.Figure()

    # Starting point
    fig.add_trace(go.Scattermapbox(
        lat=[lat], lon=[lng],
        mode="markers",
        marker=dict(size=15, color="blue"),
        text=[f"AQI: {aqi_data['aqi']} ({aqi_data['category']})"],
        name="Start"
    ))

    # Routes
    colors = ["green", "purple", "orange"]
    for i, route in enumerate(routes):
        lats = [c[0] for c in route["coordinates"]]
        lons = [c[1] for c in route["coordinates"]]
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=4),
            name=f"Route {i+1} ({route['distance']:.1f} km, Score {route['score']:.1f})"
        ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=lat, lon=lng), zoom=12),
        height=600
    )
    return fig

# -------------------------------
# Main Logic
# -------------------------------
def process_route_generation(location, distance, fitness_level, health_conditions, time_preference):
    coords = geocode_address(location)
    if not coords:
        st.error("❌ Could not geocode location.")
        return False

    aqi_data = get_air_quality_data(coords['lat'], coords['lng'])
    elevation = get_elevation(coords['lat'], coords['lng'])
    timezone = get_timezone(coords['lat'], coords['lng'])

    G, jogging_edges = score_jogging_paths(location)
    routes = generate_routes(G, jogging_edges, coords['lat'], coords['lng'])

    map_obj = create_plotly_map(coords, aqi_data, routes)

    st.session_state.route_data = {
        'coords': coords,
        'aqi_data': aqi_data,
        'elevation': elevation,
        'timezone': timezone,
        'routes': routes,
        'location': location,
        'distance': distance,
        'fitness_level': fitness_level,
        'health_conditions': health_conditions,
        'time_preference': time_preference
    }
    st.session_state.map_data = map_obj
    st.session_state.results_generated = True
    return True

def display_results():
    data = st.session_state.route_data
    st.markdown("### 📊 Environmental Metrics")
    st.metric("Air Quality Index", f"{data['aqi_data']['aqi']}", data['aqi_data']['category'])
    st.metric("Elevation", f"{data['elevation']:.1f}m")
    st.metric("Local Time", datetime.now(pytz.timezone(data['timezone'])).strftime("%H:%M"))
    st.metric("Dominant Pollutant", data['aqi_data']['dominant_pollutant'])

    st.markdown("### 🗺️ Interactive Pollution Map & Routes")
    st.plotly_chart(st.session_state.map_data, use_container_width=True)

    st.markdown("### 📊 Route Analysis")
    for i, route in enumerate(data['routes']):
        st.write(f"**Route {i
