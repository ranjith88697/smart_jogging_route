# -*- coding: utf-8 -*-
import osmnx as ox
import networkx as nx
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import pytz
import plotly.graph_objects as go

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Smart Jogging Routes", page_icon="🏃‍♂️", layout="wide")

# -------------------------------
# Helper functions
# -------------------------------
def geocode_address(address, api_key):
    """Geocode an address using Google Maps API"""
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                return {"lat": loc["lat"], "lng": loc["lng"]}
            else:
                st.error("❌ Could not geocode location. No results found.")
        else:
            st.error(f"❌ Geocoding failed: {r.status_code} - {r.text}")
    except Exception as e:
        st.error(f"❌ Geocoding error: {e}")
    return None

def get_air_quality_data(lat, lng, api_key):
    """Fetch AQI data from Google Air Quality API"""
    url = f"https://airquality.googleapis.com/v1/currentConditions:lookup?key={api_key}"
    payload = {"location": {"latitude": lat, "longitude": lng}, "languageCode": "en"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200 and "currentConditions" in r.json():
            aqi_info = r.json()["currentConditions"]["indexes"][0]
            return {
                "aqi": aqi_info.get("aqi"),
                "category": aqi_info.get("category"),
                "dominant_pollutant": aqi_info.get("dominantPollutant", "unknown")
            }
    except Exception as e:
        st.warning(f"AQI API failed: {e}")
    return {"aqi": 85, "category": "Moderate", "dominant_pollutant": "PM2.5"}

def create_map(lat, lng, routes, aqi_data):
    """Create a Plotly map with routes and AQI info"""
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

#api_key = st.secrets.get("GOOGLE_API_KEY", "")
api_key = "AIzaSyAX3v9OSj4Fg3Ad649BIRR13B09CidYNqc"

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
        # Dummy routes for demo
        routes = [
            {"coordinates": [(coords["lat"], coords["lng"]), (coords["lat"]+0.01, coords["lng"]+0.01)], "distance": 5.0, "avg_aqi": 80},
            {"coordinates": [(coords["lat"], coords["lng"]), (coords["lat"]+0.02, coords["lng"]+0.02)], "distance": 7.0, "avg_aqi": 90}
        ]
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
