# -*- coding: utf-8 -*-
import osmnx as ox
import networkx as nx
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import geopandas as gpd
import folium
from folium import Choropleth
import branca.colormap as cm
import requests
import streamlit as st # Import streamlit for st.warning and st.cache_data
#api_key = userdata.get('GOOGLE_API_KEY') # Get the API key from userdata
api_key = "AIzaSyAX3v9OSj4Fg3Ad649BIRR13B09CidYNqc"
address = "Riga, Latvia"
url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"

response = requests.get(url)
location = response.json()['results'][0]['geometry']['location']
print(f"Latitude: {location['lat']}, Longitude: {location['lng']}")

st.set_page_config(
    page_title="Smart Jogging Routes",
    page_icon="🏃‍♂️",
    layout="wide"
)


if 'results_generated' not in st.session_state:
    st.session_state.results_generated = False
if 'route_data' not in st.session_state:
    st.session_state.route_data = None
if 'map_data' not in st.session_state:
    st.session_state.map_data = None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4285f4 0%, #34a853 50%, #ea4335 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .alert-good {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .alert-moderate {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .alert-poor {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4285f4, #34a853);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

#riga_jogging_paths file retieval
# Define the area of interest
place_name = "Riga, Latvia"

# Download the walkable street network
G = ox.graph_from_place(place_name, network_type='walk')

# Convert graph to GeoDataFrame
nodes, edges = ox.graph_to_gdfs(G)

# Ensure 'surface' column exists before filtering
if 'surface' not in edges.columns:
    edges['surface'] = None # Or some other default value

# Filter jogging-friendly paths (e.g., footways, paths, cycleways)
jogging_edges = edges[edges['highway'].isin(['footway', 'path', 'cycleway'])].copy()


# Save for inspection
jogging_edges.to_file("riga_jogging_paths.geojson", driver="GeoJSON")
print(f"Extracted {len(jogging_edges)} jogging path segments.")

#Generate score and extract riga_scored_jogging_paths
# Create synthetic features
jogging_edges['length'] = jogging_edges['length'].fillna(0)

# Check if 'surface' column exists after filtering, if not create it
if 'surface' not in jogging_edges.columns:
    jogging_edges['surface'] = 'unknown'

jogging_edges['surface'] = jogging_edges['surface'].fillna('unknown')

# Encode surface type
jogging_edges['surface_encoded'] = pd.factorize(jogging_edges['surface'])[0]

# Create a synthetic target score (e.g., longer + paved = better)
jogging_edges['score'] = (
    0.5 * jogging_edges['length'] +
    10 * (jogging_edges['surface'] == 'paved').astype(int)
)

# Train a simple model
X = jogging_edges[['length', 'surface_encoded']]
y = jogging_edges['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict scores
jogging_edges['predicted_score'] = model.predict(X)

# Save scored paths
jogging_edges.to_file("riga_scored_jogging_paths.geojson", driver="GeoJSON")
print("Route scoring model trained and predictions saved.")

#Extracting Air Quality Index
def get_air_quality_data(lat, lng, GOOGLE_API_KEY):
    """Fetches actual AQI from Google Air Quality API"""
    url = f"https://airquality.googleapis.com/v1/currentConditions:lookup?key={GOOGLE_API_KEY}"
    payload = {
        "location": {
            "latitude": lat,
            "longitude": lng
        },
        "languageCode": "en"
    }

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
        else:
            st.warning(f"AQI API error {response.status_code}: {response.text}")
    except Exception as e:
        st.warning(f"Air quality API failed: {e}")


    return {
        "aqi": 85,
        "category": "Moderate",
        "dominant_pollutant": "PM2.5",
        "display_name": "Fallback AQI"
    }
    aq_data = response.json()
    print(aq_data['indexes'][0]['aqi'])  # Example: AQI value

def create_interactive_map(lat, lng, routes, heatmap_df, aqi_data):
    import plotly.graph_objects as go

    def get_aqi_color(aqi):
        if aqi <= 50:
            return 'green'
        elif aqi <= 100:
            return 'yellow'
        elif aqi <= 150:
            return 'orange'
        else:
            return 'red'

    # Create a base map centered on Riga
    fig = go.Figure(go.Scattermapbox())


    # Add pollution heatmap using scatter_mapbox with density
    fig.add_trace(go.Scattermapbox(
        lat=heatmap_df['lat'],
        lon=heatmap_df['lng'],
        mode='markers',
        marker=dict(
            size=8,
            color=heatmap_df['aqi'],
            colorscale=[
                [0, 'green'],
                [0.3, 'yellow'],
                [0.6, 'orange'],
                [1, 'red']
            ],
            opacity=0.6,
            colorbar=dict(
                title="AQI"
            )
        ),
        text=[f"AQI: {aqi:.0f}" for aqi in heatmap_df['aqi']],
        hovertemplate="<b>AQI: %{marker.color:.0f}</b><br>" +
                      "Lat: %{lat:.4f}<br>" +
                      "Lon: %{lon:.4f}<extra></extra>",
        name="Air Quality Data"
    ))

    # Add center marker
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lng],
        mode='markers',
        marker=dict(
            size=15,
            color='blue',
            symbol='circle'
        ),
        text=[f"Starting Point<br>AQI: {aqi_data['aqi']} ({aqi_data['category']})"],
        hovertemplate="<b>Starting Point</b><br>" +
                      f"AQI: {aqi_data['aqi']} ({aqi_data['category']})<br>" +
                      f"Dominant Pollutant: {aqi_data['dominant_pollutant']}<extra></extra>",
        name="Starting Point"
    ))

    # Add optimized routes
    route_colors = ['blue', 'green', 'purple', 'orange']
    for i, route in enumerate(routes):
        route_lats = [coord[0] for coord in route['coordinates']]
        route_lons = [coord[1] for coord in route['coordinates']]

        fig.add_trace(go.Scattermapbox(
            lat=route_lats,
            lon=route_lons,
            mode='lines+markers',
            line=dict(
                color=route_colors[i % len(route_colors)],
                width=4
            ),
            marker=dict(
                size=8,
                color=route_colors[i % len(route_colors)]
            ),
            text=[f"Route {i+1}" for _ in route_lats],
            hovertemplate=f"<b>Route {i+1}</b><br>" +
                          f"Distance: {route['distance']:.1f} km<br>" +
                          f"Avg AQI: {route['avg_aqi']:.0f}<extra></extra>",
            name=f"Route {i+1} ({route['distance']:.1f}km)"
        ))

    # Update layout for mapbox
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lng),
            zoom=12
        ),
        showlegend=True,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Interactive Pollution Map & Jogging Routes"
    )

    return fig

def get_health_recommendations(aqi_data, user_profile):
    """Generate personalized health recommendations"""
    aqi = aqi_data['aqi']
    recommendations = []

    if aqi <= 50:
        recommendations.append("✅ Excellent conditions for outdoor exercise!")
        alert_class = "alert-good"
    elif aqi <= 100:
        recommendations.append("⚠️ Moderate conditions. Consider shorter routes if sensitive.")
        alert_class = "alert-moderate"
    elif aqi <= 150:
        recommendations.append("🚨 Unhealthy for sensitive groups. Consider indoor alternatives.")
        alert_class = "alert-poor"
    else:
        recommendations.append("🚨 Poor air quality. Avoid outdoor exercise.")
        alert_class = "alert-poor"


    if 'Asthma' in user_profile['health_conditions']:
        if aqi > 100:
            recommendations.append("🫁 Asthma Alert: Air quality may trigger symptoms. Consider indoor exercise.")

    if 'Heart Condition' in user_profile['health_conditions']:
        if aqi > 100:
            recommendations.append("❤️ Heart Health: Poor air quality may strain cardiovascular system. Consult doctor.")


    if user_profile['fitness_level'] == 'Beginner' and aqi > 100:
        recommendations.append("🏃‍♀️ Beginner Tip: Start with indoor exercise when air quality is moderate or poor.")

    return recommendations, alert_class

def process_route_generation(location, distance, fitness_level, health_conditions, time_preference):
    """Process route generation and store results in session state"""
    try:

        coords = geocode_address(location)
        if not coords:
            st.error("❌ Could not find the specified location. Please try again.")
            return False

        # Get environmental data
        with st.status("Fetching environmental data...") as status:
            status.update(label="Getting air quality data...", state="running")
            aqi_data = get_air_quality_data(coords['lat'], coords['lng'])

            status.update(label="Getting elevation data...", state="running")
            elevation = get_elevation(coords['lat'], coords['lng'])

            status.update(label="Getting timezone data...", state="running")
            timezone = get_timezone(coords['lat'], coords['lng'])

            status.update(label="Generating pollution heatmap...", state="running")
            heatmap_data = generate_pollution_heatmap_data(coords['lat'], coords['lng'])

            status.update(label="Optimizing routes...", state="running")
            routes = generate_optimized_routes(coords, heatmap_data)

            status.update(label="Creating interactive map...", state="running")
            map_obj = create_plotly_map(coords, aqi_data, heatmap_data, routes)

            status.update(label="Complete!", state="complete")


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

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def display_results():
    """Display the generated results from session state"""
    if not st.session_state.results_generated or not st.session_state.route_data:
        return

    data = st.session_state.route_data

    # Display key metrics
    st.markdown("### 📊 Environmental Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        aqi_color = "🟢" if data['aqi_data']['aqi'] < 100 else "🟡" if data['aqi_data']['aqi'] < 150 else "🔴"
        st.metric("Air Quality Index", f"{aqi_color} {data['aqi_data']['aqi']}", data['aqi_data']['category'])

    with col2:
        st.metric("Elevation", f"{data['elevation']:.1f}m", "Above sea level")

    with col3:
        local_time = datetime.now(pytz.timezone(data['timezone']))
        st.metric("Local Time", local_time.strftime("%H:%M"), data['timezone'].split('/')[-1])

    with col4:
        st.metric("Dominant Pollutant", data['aqi_data']['dominant_pollutant'], "Primary concern")

    # Health recommendations
    user_profile = {
        'fitness_level': data['fitness_level'],
        'health_conditions': data['health_conditions'],
        'time_preference': data['time_preference']
    }

    recommendations, alert_class = get_health_recommendations(data['aqi_data'], user_profile)

    st.markdown("### 🏥 Health Recommendations")
    for rec in recommendations:
        st.markdown(f'<div class="{alert_class}">{rec}</div>', unsafe_allow_html=True)

    # Display Plotly map
    st.markdown("### 🗺️ Interactive Pollution Map & Routes")
    if st.session_state.map_data:
        st.plotly_chart(st.session_state.map_data, use_container_width=True)


    st.markdown("### 🗺️ Still working on Heat Maps of Polluted areas")
    try:
        pydeck_map = create_alternative_map_with_pydeck(data['coords'], data['aqi_data'],
                                                       generate_pollution_heatmap_data(data['coords']['lat'], data['coords']['lng']),
                                                       data['routes'])
        st.pydeck_chart(pydeck_map)
    except ImportError:
        st.info("PyDeck not available. Install with: pip install pydeck")

    # Route analysis
    st.markdown("### 📊 Route Analysis")
    for i, route in enumerate(data['routes']):
        with st.expander(f"Route {i+1}: {route['distance']:.1f}km (Avg AQI: {route['avg_aqi']:.0f})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Distance:** {route['distance']:.1f} km")
                st.write(f"**Estimated Time:** {route['distance'] * 6:.0f} minutes")
                st.write(f"**Average AQI:** {route['avg_aqi']:.0f}")
            with col2:
                quality = "Excellent" if route['avg_aqi'] < 50 else "Good" if route['avg_aqi'] < 100 else "Moderate"
                st.write(f"**Air Quality:** {quality}")
                st.write(f"**Calories (est.):** {route['distance'] * 65:.0f}")
                st.write(f"**Difficulty:** {data['fitness_level']}")


    st.markdown("### 📈 Environmental Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌡️ Current Conditions**")
        st.write(f"• Primary pollutant: {data['aqi_data']['dominant_pollutant']}")
        st.write(f"• Air quality category: {data['aqi_data']['category']}")
        st.write(f"• Elevation impact: {'Minimal' if data['elevation'] < 100 else 'Moderate' if data['elevation'] < 500 else 'Significant'}")

    with col2:
        st.markdown("**🎯 Optimization Results**")
        best_route = min(data['routes'], key=lambda x: x['avg_aqi'])
        worst_route = max(data['routes'], key=lambda x: x['avg_aqi'])
        improvement = ((worst_route['avg_aqi'] - best_route['avg_aqi']) / worst_route['avg_aqi'] * 100) if worst_route['avg_aqi'] > 0 else 0
        st.write(f"• Best route AQI: {best_route['avg_aqi']:.0f}")
        st.write(f"• Pollution avoidance: {improvement:.0f}%")
        st.write(f"• Health risk: {'Low' if best_route['avg_aqi'] < 100 else 'Moderate' if best_route['avg_aqi'] < 150 else 'High'}")

def main():

    with open("animated_header.html", "r", encoding="utf-8") as f:
        animated_header = f.read()
        st.markdown(animated_header, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🏃‍♂️ Smart Air Quality Jogging Routes</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time pollution analysis for optimal jogging routes")


    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            width: 400px !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 400px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    with st.sidebar:
        st.header("🎯 Route Preferences")
        location = st.text_input("📍 Starting Location", "Times Square, New York", key="location_input")
        distance = st.slider("🏁 Target Distance (km)", 1.0, 15.0, 5.0, 0.5, key="distance_slider")
        fitness_level = st.selectbox("💪 Fitness Level", ["Beginner", "Intermediate", "Advanced"], key="fitness_select")
        health_conditions = st.multiselect("🏥 Health Considerations",
                                         ["Asthma", "Heart Condition", "None"],
                                         default=["None"], key="health_select")
        time_preference = st.selectbox("🕐 Preferred Time",
                                     ["Early Morning", "Morning", "Evening", "Night"], key="time_select")

        # Generate route button
        if st.button("🗺️ Generate Smart Routes", type="primary", key="generate_button"):
            success = process_route_generation(location, distance, fitness_level, health_conditions, time_preference)
            if success:
                st.success("✅ Routes generated successfully!")

        if st.session_state.results_generated:
            if st.button("🗑️ Clear Results", key="clear_button"):
                st.session_state.results_generated = False
                st.session_state.route_data = None
                st.session_state.map_data = None

    # Main content area
    if st.session_state.results_generated:
        display_results()
    else:
        # Welcome message when no results
        st.markdown("""
        ## Welcome to Smart Jogging Routes! 🏃‍♂️

        This app provides intelligent route recommendations based on:
        - **Real-time air quality data** from Google Air Quality API
        - **Interactive pollution visualization** with Plotly maps
        - **Health-conscious route optimization**
        - **Personalized recommendations** based on your fitness level and health conditions

        Simply enter your preferences in the sidebar and click "Generate Smart Routes" to get started!

        ---
        *Powered by Google Maps APIs and advanced environmental data analysis.*
        """)

if __name__ == "__main__":
    main()
