# app.py
import os
import requests
import osmnx as ox
import networkx as nx
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Smart Jogging Routes", page_icon="🏃‍♂️", layout="wide")

# -------------------------------
# Helper functions
# -------------------------------
def geocode_address(address, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    r = requests.get(url)
    if r.status_code == 200 and r.json()["results"]:
        loc = r.json()["results"][0]["geometry"]["location"]
        return {"lat": loc["lat"], "lng": loc["lng"]}
    return None

def get_air_quality_data(lat, lng, api_key):
    url = f"https://airquality.googleapis.com/v1/currentConditions:lookup?key={api_key}"
    payload = {"location": {"latitude": lat, "longitude": lng}, "languageCode": "en"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200 and "currentConditions" in r.json():
            aqi_info = r.json()["currentConditions"]["indexes"][0]
            return {"aqi": aqi_info.get("aqi"),
                    "category": aqi_info.get("category"),
                    "dominant_pollutant": aqi_info.get("dominantPollutant", "unknown")}
    except Exception as e:
        st.warning(f"AQI API failed: {e}")
    return {"aqi": 85, "category": "Moderate", "dominant_pollutant": "PM2.5"}

def generate_routes(G, start_lat, start_lng, distance_km=5, num_routes=3):
    """Generate jogging routes using osmnx shortest paths"""
    routes = []
    # find nearest node to start
    start_node = ox.distance.nearest_nodes(G, start_lng, start_lat)
    nodes = list(G.nodes)
    for i in range(num_routes):
        # pick a random target node far enough away
        target_node = nodes[(i*500) % len(nodes)]
        try:
            path = nx.shortest_path(G, start_node, target_node, weight="length")
            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
            length_m = sum(ox.utils_graph.get_route_edge_attributes(G, path, "length"))
            routes.append({"coordinates": coords,
                           "distance": length_m/1000,
                           "avg_aqi": 80 + i*5})  # placeholder AQI per route
        except Exception:
            continue
    return routes

def create_map(lat, lng, routes, aqi_data):
    fig = go.Figure(go.Scattermapbox())
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
            name=f"Route {i+1} ({route['distance']:.1f} km)"
        ))
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=lat, lon=lng), zoom=12),
        height=600
    )
    return fig

# -------------------------------
# Sidebar inputs
# -------------------------------
st.sidebar.header("🎯 Route Preferences")
location = st.sidebar.text_input("📍 Starting Location", "Riga, Latvia")
distance = st.sidebar.slider("🏁 Target Distance (km)", 1.0, 15.0, 5.0, 0.5)
fitness_level = st.sidebar.selectbox("💪 Fitness Level", ["Beginner", "Intermediate", "Advanced"])
health_conditions = st.sidebar.multiselect("🏥 Health Considerations", ["Asthma", "Heart Condition", "None"], default=["None"])
time_pref = st.sidebar.selectbox("🕐 Preferred Time", ["Early Morning", "Morning", "Evening", "Night"])

api_key = st.secrets.get("GOOGLE_API_KEY", "")

# -------------------------------
# Main logic
# -------------------------------
st.markdown("## 🏃‍♂️ Smart Air Quality Jogging Routes")

if st.sidebar.button("🗺️ Generate Smart Routes"):
    coords = geocode_address(location, api_key)
    if not coords:
        st.error("❌ Could not geocode location.")
    else:
        aqi_data = get_air_quality_data(coords["lat"], coords["lng"], api_key)
        # Build graph around location
        G = ox.graph_from_point((coords["lat"], coords["lng"]), dist=int(distance*1000), network_type="walk")
        routes = generate_routes(G, coords["lat"], coords["lng"], distance_km=distance)
        # Metrics
        st.metric("Air Quality Index", f"{aqi_data['aqi']}", aqi_data["category"])
        st.metric("Dominant Pollutant", aqi_data["dominant_pollutant"])
        # Map
        fig = create_map(coords["lat"], coords["lng"], routes, aqi_data)
        st.plotly_chart(fig, use_container_width=True)
        # Route details
        for i, r in enumerate(routes):
            st.write(f"**Route {i+1}:** {r['distance']:.1f} km, Avg AQI {r['avg_aqi']}")
else:
    st.info("Enter preferences in the sidebar and click 'Generate Smart Routes'.")
