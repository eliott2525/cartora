import pandas as pd
import numpy as np
import folium
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from operator_distances import load_data, haversine_distance
import os
from tqdm import tqdm

def create_operator_map(data, operator, output_dir='outputs'):
    """Create an interactive map for a specific operator's antennas."""
    # Filter data for operator
    operator_data = data[data['Exploitant'] == operator]
    
    # Create base map centered on France
    m = folium.Map(
        location=[46.2276, 2.2137],  # Center of France
        zoom_start=6,
        tiles='cartodbpositron'
    )
    
    # Add antenna locations
    for _, row in operator_data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color='red',
            fill=True,
            popup=f"Antenna ID: {row['Numéro de support']}"
        ).add_to(m)
    
    # Add heatmap layer
    locations = operator_data[['Latitude', 'Longitude']].values.tolist()
    plugins.HeatMap(locations).add_to(m)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save map
    m.save(os.path.join(output_dir, f'coverage_map_{operator.lower().replace(" ", "_")}.html'))

def create_density_heatmap(data, output_dir='outputs'):
    """Create a static heatmap showing antenna density across France."""
    plt.figure(figsize=(15, 10))
    
    # France boundaries
    france_bounds = {
        'lat_min': 47,
        'lat_max': 49.5,
        'lon_min': 1,
        'lon_max': 4.5
    }
    
    # Create grid for density calculation
    xmin, xmax = france_bounds['lon_min'], france_bounds['lon_max']
    ymin, ymax = france_bounds['lat_min'], france_bounds['lat_max']
    
    # Calculate density for each operator
    for operator in data['Exploitant'].unique():
        plt.figure(figsize=(15, 10))
        operator_data = data[data['Exploitant'] == operator]
        
        x = operator_data['Longitude'].values
        y = operator_data['Latitude'].values
        
        # Calculate the point density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # Sort the points by density
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        
        # Plot
        plt.scatter(x, y, c=z, s=50, alpha=0.5, cmap='viridis')
        plt.colorbar(label='Antenna Density')
        
        # Set plot bounds to France
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        
        plt.title(f'Antenna Density - {operator}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'density_map_{operator.lower().replace(" ", "_")}.png'))
        plt.close()

def identify_low_coverage_areas(data, grid_size=0.5, threshold_percentile=10, output_dir='outputs'):
    """Identify areas with low antenna coverage."""
    # Create grid over France
    france_bounds = {
        'lat_min': 47,
        'lat_max': 49.5,
        'lon_min': 1,
        'lon_max': 4.5
    }
    
    lon_edges = np.arange(france_bounds['lon_min'], france_bounds['lon_max'], grid_size)
    lat_edges = np.arange(france_bounds['lat_min'], france_bounds['lat_max'], grid_size)
    
    for operator in data['Exploitant'].unique():
        plt.figure(figsize=(15, 10))
        operator_data = data[data['Exploitant'] == operator]
        
        # Calculate antenna count in each grid cell
        H, _, _ = np.histogram2d(
            operator_data['Latitude'],
            operator_data['Longitude'],
            bins=[lat_edges, lon_edges]
        )
        
        # Identify low coverage areas (below threshold)
        threshold = np.percentile(H[H > 0], threshold_percentile)
        low_coverage = np.ma.masked_where(H > threshold, H)
        
        # Plot
        plt.imshow(
            low_coverage,
            extent=[france_bounds['lon_min'], france_bounds['lon_max'],
                   france_bounds['lat_min'], france_bounds['lat_max']],
            origin='lower',
            cmap='Reds_r',
            aspect='auto'
        )
        
        plt.colorbar(label='Antenna Count')
        plt.title(f'Low Coverage Areas - {operator}\n(Red indicates fewer antennas)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'low_coverage_{operator.lower().replace(" ", "_")}.png'))
        plt.close()

def create_comparative_analysis(data, output_dir='outputs'):
    """Create comparative visualizations of coverage between operators."""
    # Prepare data for box plot
    distances_by_operator = []
    operators = []
    
    for operator in data['Exploitant'].unique():
        operator_data = data[data['Exploitant'] == operator]
        coords = operator_data[['Latitude', 'Longitude']].values
        
        # Calculate distances to nearest antenna for each point
        for i in range(len(coords)):
            current = coords[i]
            others = np.delete(coords, i, axis=0)
            if len(others) > 0:
                distances = np.array([
                    haversine_distance(current[0], current[1], other[0], other[1])
                    for other in others
                ])
                min_distance = np.min(distances)
                distances_by_operator.append(min_distance)
                operators.append(operator)
    
    # Create box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=operators, y=distances_by_operator)
    plt.title('Distribution of Distances to Nearest Antenna by Operator')
    plt.xlabel('Operator')
    plt.ylabel('Distance to Nearest Antenna (km)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_distribution_comparison.png'))
    plt.close()

def main():
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Create individual operator maps
    print("\nCreating operator maps...")
    for operator in tqdm(data['Exploitant'].unique()):
        create_operator_map(data, operator, output_dir)
    
    # Create density heatmaps
    print("\nCreating density heatmaps...")
    create_density_heatmap(data, output_dir)
    
    # Identify low coverage areas
    print("\nIdentifying low coverage areas...")
    identify_low_coverage_areas(data, output_dir=output_dir)
    
    # Create comparative analysis
    print("\nCreating comparative analysis...")
    create_comparative_analysis(data, output_dir)
    
    print("\nAnalysis complete! Check the 'outputs' directory for results.")

if __name__ == "__main__":
    main() 