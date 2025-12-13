import streamlit as st
import googlemaps
import requests
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import json
import time
import osmnx as ox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import networkx as nx

GOOGLE_API_KEY = "AIzaSyAX3v9OSj4Fg3Ad649BIRR13B09CidYNqc"

gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

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

# Utility Functions
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


@st.cache_data(ttl=3600)
def score_jogging_paths_by_coords(lat, lng, dist_m=3000):
    """Build walk graph around coordinates & score jogging paths"""

    G = ox.graph_from_point(
        (lat, lng),
        dist=dist_m,
        network_type='walk'
    )

    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()

    if 'surface' not in edges.columns:
        edges['surface'] = 'unknown'

    jogging_edges = edges[
        edges['highway'].isin(['footway', 'path', 'cycleway'])
    ].copy()

    jogging_edges['length'] = jogging_edges['length'].fillna(0)
    jogging_edges['surface'] = jogging_edges['surface'].fillna('unknown')
    jogging_edges['surface_encoded'] = pd.factorize(jogging_edges['surface'])[0]

    jogging_edges['score'] = (
        0.6 * jogging_edges['length'] +
        15 * (jogging_edges['surface'] == 'paved').astype(int)
    )

    X = jogging_edges[['length', 'surface_encoded']]
    y = jogging_edges['score']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    jogging_edges['predicted_score'] = model.predict(X)

    return G, jogging_edges

def generate_ml_optimized_routes(center_coords, heatmap_data, max_routes=3):
    G, jogging_edges = score_jogging_paths_by_coords(
        center_coords['lat'],
        center_coords['lng']
    )

    G_nx = nx.Graph(G)

    center_node = ox.distance.nearest_nodes(
        G_nx,
        center_coords['lng'],
        center_coords['lat']
    )

    routes = []
    sampled_edges = jogging_edges.sample(
        n=min(25, len(jogging_edges)), random_state=42
    )

    for _, edge in sampled_edges.iterrows():
        try:
            path = nx.shortest_path(
                G_nx,
                center_node,
                edge['u'],
                weight='length'
            )

            coords = [(G_nx.nodes[n]['y'], G_nx.nodes[n]['x']) for n in path]

            distance_km = edge['length'] / 1000
            avg_aqi = np.mean([p['aqi'] for p in heatmap_data[:10]])

            routes.append({
                'coordinates': coords,
                'distance': distance_km,
                'avg_aqi': avg_aqi,
                'predicted_score': edge['predicted_score'],
                'route_id': len(routes) + 1
            })

            if len(routes) >= max_routes:
                break

        except Exception:
            continue

    return sorted(
        routes,
        key=lambda x: (-x['predicted_score'], x['avg_aqi'])
    )

def generate_loop_routes(center_coords, target_distance_km, heatmap_data, max_routes=3, tolerance=0.25):
    """Generate jogging loop routes around a center coordinate with guaranteed output."""
    try:
        G, jogging_edges = score_jogging_paths_by_coords(center_coords['lat'], center_coords['lng'], dist_m=3000)
        G_nx = nx.Graph(G)
        start_node = ox.distance.nearest_nodes(G_nx, center_coords['lng'], center_coords['lat'])
    except Exception:
        # Fallback graph empty
        G_nx = None
        start_node = None

    routes = []

    # Normal route generation
    if G_nx is not None and jogging_edges is not None and len(jogging_edges) > 0:
        candidate_nodes = list(G_nx.nodes)
        np.random.shuffle(candidate_nodes)
        for mid_node in candidate_nodes[:300]:
            try:
                path_out = nx.shortest_path(G_nx, start_node, mid_node, weight='length')
                path_back = nx.shortest_path(G_nx, mid_node, start_node, weight='length')
                loop_path = path_out + path_back[1:]

                total_length_m = sum(
                    list(G_nx.get_edge_data(u, v).values())[0].get('length', 0)
                    for u, v in zip(loop_path[:-1], loop_path[1:])
                )
                total_km = total_length_m / 1000
                min_dist = target_distance_km * (1 - tolerance)
                max_dist = target_distance_km * (1 + tolerance)
                if not (min_dist <= total_km <= max_dist):
                    continue

                coords = [(G_nx.nodes[n]['y'], G_nx.nodes[n]['x']) for n in loop_path]
                avg_aqi = np.mean([p['aqi'] for p in heatmap_data[:10]])
                predicted_score = jogging_edges['predicted_score'].sample(n=min(5, len(jogging_edges))).mean()

                routes.append({
                    'coordinates': coords,
                    'distance': total_km,
                    'avg_aqi': avg_aqi,
                    'predicted_score': predicted_score,
                    'route_id': len(routes) + 1,
                    'loop': True
                })
                if len(routes) >= max_routes:
                    break
            except Exception:
                continue

    # Fallback: simple synthetic loop around start point
    if not routes:
        lat, lng = center_coords['lat'], center_coords['lng']
        coords = [
            (lat, lng),
            (lat + 0.002, lng + 0.002),
            (lat + 0.003, lng - 0.001),
            (lat, lng)
        ]
        total_distance = sum(geodesic(coords[i], coords[i+1]).km for i in range(len(coords)-1))
        avg_aqi = np.mean([p['aqi'] for p in heatmap_data[:10]])
        routes.append({
            'coordinates': coords,
            'distance': total_distance,
            'avg_aqi': avg_aqi,
            'predicted_score': 50,  # default
            'route_id': 1,
            'loop': True
        })

    return routes


@st.cache_data(ttl=600) 
def generate_pollution_heatmap_data(center_lat, center_lng, radius_km=5):
    """Generate grid points for pollution heatmap"""
    points = []
    grid_size = 20  
    
    # Calculate bounds
    lat_range = radius_km / 111.0  # Approximate km to degrees
    lng_range = radius_km / (111.0 * np.cos(np.radians(center_lat)))
    
    for i in range(grid_size):
        for j in range(grid_size):
            lat = center_lat + (i - grid_size/2) * lat_range / (grid_size/2)
            lng = center_lng + (j - grid_size/2) * lng_range / (grid_size/2)
            
            # Simulate AQI based on distance from center and some randomness
            distance = geodesic((center_lat, center_lng), (lat, lng)).kilometers
            
            # More realistic pollution simulation
            base_pollution = 60  
            distance_factor = max(0, distance * 8) 
            random_factor = np.random.normal(0, 15)  
            
            simulated_aqi = max(30, min(250, base_pollution + distance_factor + random_factor))
            
            points.append({
                'lat': lat,
                'lng': lng,
                'aqi': simulated_aqi,
                'weight': min(1.0, simulated_aqi / 200.0)  
            })
    
    return points

def create_plotly_map(center_coords, aqi_data, heatmap_data, routes):
    """Create an interactive Plotly map with pollution heatmap and routes"""
    lat, lng = center_coords['lat'], center_coords['lng']
    
   
    fig = go.Figure()
    
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    
    def get_aqi_color(aqi):
        if aqi <= 50:
            return 'green'
        elif aqi <= 100:
            return 'yellow'
        elif aqi <= 150:
            return 'orange'
        else:
            return 'red'
    
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
            #hovertemplate=f"<b>Route {i+1}</b><br>" +
             #             f"Distance: {route['distance']:.1f} km<br>" +
              #            f"Avg AQI: {route['avg_aqi']:.0f}<extra></extra>",
            #name=f"Route {i+1} ({route['distance']:.1f}km)"
            hovertemplate=f"<b>Route {i+1}</b><br>" +
              f"Distance: {route['distance']:.1f} km<br>" +
              f"Avg AQI: {route['avg_aqi']:.0f}<br>" +
              f"Jogging Score: {route['predicted_score']:.1f}<extra></extra>"
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

def create_alternative_map_with_pydeck(center_coords, aqi_data, heatmap_data, routes):
    """Alternative map using PyDeck (uncomment if you prefer this option)"""
    import pydeck as pdk
    
    lat, lng = center_coords['lat'], center_coords['lng']
    
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df['weight'] = heatmap_df['aqi'] / 200.0
    
   
    layers = []
    
    # Heatmap layer
    layers.append(pdk.Layer(
        'HeatmapLayer',
        data=heatmap_df,
        get_position=['lng', 'lat'],
        get_weight='weight',
        radius_pixels=60,
        intensity=1,
        threshold=0.03,
        get_fill_color=[255, 0, 0, 140]
    ))
    
    # Starting point layer
    start_point_df = pd.DataFrame([{
        'lat': lat,
        'lng': lng,
        'size': 100,
        'color': [0, 0, 255, 160]
    }])
    
    layers.append(pdk.Layer(
        'ScatterplotLayer',
        data=start_point_df,
        get_position=['lng', 'lat'],
        get_radius='size',
        get_fill_color='color',
        pickable=True
    ))
    
    # Route layers
    route_colors = [[0, 0, 255, 200], [0, 255, 0, 200], [255, 0, 255, 200]]
    for i, route in enumerate(routes[:3]):  # Limit to 3 routes
        route_df = pd.DataFrame([
            {'lat': coord[0], 'lng': coord[1]} for coord in route['coordinates']
        ])
        
        layers.append(pdk.Layer(
            'PathLayer',
            data=[{
                'path': [[coord[1], coord[0]] for coord in route['coordinates']],
                'color': route_colors[i % len(route_colors)],
                'width': 5
            }],
            get_path='path',
            get_color='color',
            get_width='width',
            width_scale=1,
            pickable=True
        ))
    
    
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lng,
        zoom=12,
        pitch=0
    )
    
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            'text': 'AQI: {aqi}\nLat: {lat:.4f}\nLon: {lng:.4f}'
        }
    )
    
    return deck

@st.cache_data(ttl=600)
def generate_optimized_routes(center_coords, heatmap_data, num_routes=3):
    """Generate optimized jogging routes avoiding high pollution areas"""
    routes = []
    lat, lng = center_coords['lat'], center_coords['lng']
    
    # Create pollution grid for route optimization
    pollution_grid = {}
    for point in heatmap_data:
        key = (round(point['lat'], 4), round(point['lng'], 4))
        pollution_grid[key] = point['aqi']
    
    # Generate different route patterns
    route_patterns = [
        
        [(lat, lng), (lat + 0.008, lng + 0.008), (lat + 0.015, lng + 0.005), 
         (lat + 0.018, lng - 0.005), (lat + 0.008, lng - 0.008), (lat, lng)],
         
        [(lat, lng), (lat - 0.008, lng - 0.008), (lat - 0.015, lng - 0.005), 
         (lat - 0.018, lng + 0.005), (lat - 0.008, lng + 0.008), (lat, lng)],
        
        [(lat, lng), (lat + 0.005, lng + 0.015), (lat + 0.010, lng + 0.020), 
         (lat + 0.005, lng + 0.025), (lat - 0.005, lng + 0.015), (lat, lng)]
    ]
    
    for i, pattern in enumerate(route_patterns):
        coordinates = pattern
        
       
        total_distance = sum(geodesic(coordinates[j], coordinates[j+1]).kilometers 
                           for j in range(len(coordinates)-1))
        
        
        aqi_values = []
        for coord in coordinates:
            key = (round(coord[0], 4), round(coord[1], 4))
            aqi = pollution_grid.get(key, 80)  
            aqi_values.append(aqi)
        
        avg_aqi = np.mean(aqi_values)
        
        routes.append({
            'coordinates': coordinates,
            'distance': total_distance,
            'avg_aqi': avg_aqi,
            'route_id': i + 1
        })
    
    return sorted(routes, key=lambda x: x['avg_aqi'])  

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
        # Geocode the user location
        coords = geocode_address(location)
        if not coords:
            st.error("❌ Could not find the specified location. Please try again.")
            return False

        st.write("DEBUG coords:", coords, type(coords))

        # Fetch environmental data and generate routes with a spinner
        with st.spinner("Fetching environmental data and generating routes..."):
            aqi_data = get_air_quality_data(coords['lat'], coords['lng'])
            elevation = get_elevation(coords['lat'], coords['lng'])
            timezone = get_timezone(coords['lat'], coords['lng'])
            heatmap_data = generate_pollution_heatmap_data(coords['lat'], coords['lng'])

            routes = generate_loop_routes(
                center_coords=coords,
                target_distance_km=distance,
                heatmap_data=heatmap_data
            )

            map_obj = create_plotly_map(coords, aqi_data, heatmap_data, routes)

        # Store results in session state
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

    # ✅ HARD GUARD — Prevent crash if no routes
    if not data.get('routes'):
        st.warning(
            "⚠️ No jogging loop routes could be generated for the selected distance.\n\n"
            "Try:\n"
            "• Increasing the target distance\n"
            "• Choosing a denser area\n"
            "• Trying again (routes are stochastic)"
        )
        return  # Exit early if no routes

    # Display key metrics
    st.markdown("### 📊 Environmental Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        aqi_color = "🟢" if data['aqi_data']['aqi'] <= 50 else "🟡" if data['aqi_data']['aqi'] <= 100 else "🔴"
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
    
    # Route Analysis
    st.markdown("### 📊 Route Analysis")
    for i, route in enumerate(data['routes']):
        with st.expander(
            f"Route {i+1}: {route['distance']:.2f} km "
            f"(Score: {route.get('predicted_score', 0):.1f})"
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Distance:** {route['distance']:.2f} km")
                st.write(f"**Average AQI:** {route['avg_aqi']:.0f}")
            with col2:
                route_type = "Loop 🔁" if route.get('loop', False) else "Point-to-Point"
                st.write(f"**Route Type:** {route_type}")
                st.write(f"**Jogging Score:** {route.get('predicted_score', 0):.1f}")
    
    # Optimization Insights
    st.markdown("### 📈 Environmental Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌡️ Current Conditions**")
        st.write(f"• Primary pollutant: {data['aqi_data']['dominant_pollutant']}")
        st.write(f"• Air quality category: {data['aqi_data']['category']}")
        elevation_impact = (
            "Minimal" if data['elevation'] < 100 
            else "Moderate" if data['elevation'] < 500 
            else "Significant"
        )
        st.write(f"• Elevation impact: {elevation_impact}")
    
    with col2:
        st.markdown("**🎯 Optimization Results**")
        # Safe computation of best/worst route
        if data['routes']:
            best_route = min(data['routes'], key=lambda x: x['avg_aqi'])
            worst_route = max(data['routes'], key=lambda x: x['avg_aqi'])
            improvement = (
                (worst_route['avg_aqi'] - best_route['avg_aqi']) /
                worst_route['avg_aqi'] * 100
                if worst_route['avg_aqi'] > 0 else 0
            )
            st.write(f"• Best route AQI: {best_route['avg_aqi']:.0f}")
            st.write(f"• Pollution avoidance: {improvement:.0f}%")
        else:
            st.write("• Pollution avoidance: N/A")


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
        location = st.text_input("📍 Starting Location", "Riga, Latvia", key="location_input")
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
