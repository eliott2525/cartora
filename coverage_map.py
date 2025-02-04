import pandas as pd
import folium
from folium import plugins
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(antennas_path='data/antennas.csv', locations_path='data/locations.csv'):
    """Load and merge the antenna and location data."""
    print("Loading data...")
    
    # Load the datasets
    antennas = pd.read_csv(antennas_path, delimiter=';', encoding='latin1')
    locations = pd.read_csv(locations_path, delimiter=';', encoding='latin1')
    
    # Rename the column in locations to match antennas
    locations = locations.rename(columns={'Numéro du support': 'Numéro de support'})
    
    # Merge the dataframes
    merged = pd.merge(antennas, locations, on='Numéro de support')
    
    # Convert coordinates to numeric and drop missing values
    merged['Longitude'] = pd.to_numeric(merged['Longitude'], errors='coerce')
    merged['Latitude'] = pd.to_numeric(merged['Latitude'], errors='coerce')
    merged = merged.dropna(subset=['Longitude', 'Latitude'])
    
    # Remove duplicates based on coordinates (since we only care about unique locations for coverage)
    merged = merged.drop_duplicates(subset=['Longitude', 'Latitude'])
    
    return merged

def create_coverage_map(data):
    """Create an interactive map showing antenna coverage."""
    print("Creating coverage map...")
    
    # Create a base map centered on France
    m = folium.Map(location=[46.2276, 2.2137], zoom_start=6)
    
    # Create a heatmap layer
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in data.iterrows()]
    plugins.HeatMap(heat_data, radius=15).add_to(m)
    
    # Calculate the convex hull of all points to show coverage boundary
    points = data[['Longitude', 'Latitude']].values
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Create a polygon of the convex hull
    hull_polygon = folium.Polygon(
        locations=[[p[1], p[0]] for p in hull_points],
        color='red',
        weight=2,
        fill=False,
        popup='Coverage Area'
    )
    hull_polygon.add_to(m)
    
    # Add a layer control
    folium.LayerControl().add_to(m)
    
    # Save the map
    output_path = 'coverage_map.html'
    m.save(output_path)
    print(f"Map saved to {output_path}")
    
    return m

def create_density_plot(data):
    """Create a static density plot using matplotlib."""
    print("Creating density plot...")
    
    # Create the plot with a larger size and higher DPI
    plt.figure(figsize=(20, 20), dpi=300)
    
    # Set a light background color
    plt.gca().set_facecolor('#f0f0f0')
    plt.gcf().set_facecolor('#f0f0f0')
    
    # Create a more detailed density plot
    sns.kdeplot(
        data=data,
        x='Longitude',
        y='Latitude',
        cmap='YlOrRd',  # Yellow to Orange to Red colormap
        fill=True,
        levels=30,
        alpha=0.6,
        thresh=0.05
    )
    
    # Add points with better visibility
    plt.scatter(data['Longitude'], data['Latitude'], 
               c='#1f77b4',  # Blue color
               s=2,          # Slightly larger points
               alpha=0.2,    # More transparency
               label='Antenna Locations')
    
    # Set the plot bounds to cover France more precisely
    plt.xlim(-5, 8.5)
    plt.ylim(42, 51.5)
    
    # Add grid for better reference
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Improve title and labels
    plt.title('Antenna Coverage Density in France', 
              fontsize=16, 
              pad=20,
              fontweight='bold')
    plt.xlabel('Longitude', fontsize=12, labelpad=10)
    plt.ylabel('Latitude', fontsize=12, labelpad=10)
    
    # Add some major cities for reference
    cities = {
        'Paris': (2.3522, 48.8566),
        'Marseille': (5.3698, 43.2965),
        'Lyon': (4.8357, 45.7640),
        'Toulouse': (1.4442, 43.6047),
        'Bordeaux': (-0.5792, 44.8378),
        'Lille': (3.0573, 50.6292),
        'Strasbourg': (7.7521, 48.5734),
    }
    
    for city, coords in cities.items():
        plt.plot(coords[0], coords[1], 'k*', markersize=10)
        plt.annotate(city, 
                    (coords[0], coords[1]), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='black')
    
    # Add a legend
    plt.legend(['Antenna Location'], loc='upper right')
    
    # Add text box with statistics
    stats_text = (f"Total Antennas: {len(data):,}\n"
                 f"Unique Locations: {len(data.drop_duplicates(['Longitude', 'Latitude'])):,}\n"
                 f"Operators: {', '.join(sorted(data['Exploitant'].unique()))}")
    
    plt.text(0.02, 0.02, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             fontsize=10,
             verticalalignment='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with high quality
    output_path = 'coverage_density.png'
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight',
                facecolor='#f0f0f0',
                edgecolor='none')
    print(f"Density plot saved to {output_path}")
    plt.close()

def main():
    # Load the data
    data = load_data()
    print(f"Loaded {len(data)} unique antenna locations")
    
    # Create both visualizations
    create_coverage_map(data)
    create_density_plot(data)
    
    # Print some statistics
    print("\nCoverage Statistics:")
    print(f"Longitude range: {data['Longitude'].min():.2f} to {data['Longitude'].max():.2f}")
    print(f"Latitude range: {data['Latitude'].min():.2f} to {data['Latitude'].max():.2f}")
    print(f"Number of unique locations: {len(data)}")
    print(f"Number of unique exploitants: {data['Exploitant'].nunique()}")
    print("\nExploitants:", ', '.join(sorted(data['Exploitant'].unique())))

if __name__ == "__main__":
    main() 