import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from itertools import combinations
from collections import defaultdict

def validate_coordinates(data):
    """Validate coordinate data and print statistics."""
    print("\nData Validation:")
    print(f"Total number of antennas: {len(data)}")
    
    # Check coordinate ranges (France mainland roughly)
    valid_lat_range = (41.0, 51.5)  # France latitude range
    valid_lon_range = (-5.0, 10.0)  # France longitude range
    
    invalid_coords = data[
        ~((data['Latitude'].between(*valid_lat_range)) & 
          (data['Longitude'].between(*valid_lon_range)))
    ]
    
    if len(invalid_coords) > 0:
        print(f"\nFound {len(invalid_coords)} antennas with coordinates outside France mainland:")
        print(invalid_coords[['Exploitant', 'Latitude', 'Longitude']].head())
    
    # Check for duplicates
    duplicates = data[data.duplicated(['Latitude', 'Longitude', 'Exploitant'], keep=False)]
    if len(duplicates) > 0:
        print(f"\nFound {len(duplicates)} duplicate entries (same coordinates and operator)")
    
    # Print operator statistics
    print("\nAntennas per operator:")
    print(data['Exploitant'].value_counts())

def load_data(antennas_path='data/antennas.csv', locations_path='data/locations.csv'):
    """Load and merge antenna data with locations."""
    # Load antennas file
    print("Loading antennas data...")
    antennas = pd.read_csv(antennas_path, delimiter=';', encoding='latin1')
    print(f"Loaded {len(antennas)} antenna records")
    
    # Load locations file
    print("Loading locations data...")
    locations = pd.read_csv(locations_path, delimiter=';', encoding='latin1')
    print(f"Loaded {len(locations)} location records")
    
    # Rename column for merging
    locations = locations.rename(columns={'Numéro du support': 'Numéro de support'})
    
    # Merge datasets
    print("Merging datasets...")
    merged = pd.merge(antennas, locations, on='Numéro de support')
    print(f"After merging: {len(merged)} records")
    
    # Select relevant columns
    merged = merged[['Numéro de support', 'Exploitant', 'Longitude', 'Latitude']]
    
    # Convert coordinates to numeric, handling both string and numeric formats
    def convert_coordinate(x):
        if isinstance(x, str):
            return float(x.replace(',', '.'))
        return float(x)
    
    # Convert coordinates safely
    merged['Longitude'] = merged['Longitude'].apply(convert_coordinate)
    merged['Latitude'] = merged['Latitude'].apply(convert_coordinate)
    
    # Remove invalid coordinates
    initial_len = len(merged)
    merged = merged.dropna(subset=['Longitude', 'Latitude'])
    if initial_len - len(merged) > 0:
        print(f"Removed {initial_len - len(merged)} records with invalid coordinates")
    
    # Remove duplicates based on coordinates and operator
    initial_len = len(merged)
    merged = merged.drop_duplicates(subset=['Latitude', 'Longitude', 'Exploitant'])
    if initial_len - len(merged) > 0:
        print(f"Removed {initial_len - len(merged)} duplicate records")
    
    return merged

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in kilometers."""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def process_operator_chunk(args):
    """Process a chunk of antennas for a specific operator."""
    operator_data, chunk_size = args
    distances = []
    
    # Convert to numpy array for faster operations
    coordinates = operator_data[['Latitude', 'Longitude']].values
    
    # Process each antenna
    for i in tqdm(range(len(coordinates)), desc="Processing antennas", leave=False):
        current_coord = coordinates[i]
        
        # Calculate distances to all other antennas
        dists = haversine_distance(
            current_coord[0], current_coord[1],
            coordinates[:, 0], coordinates[:, 1]
        )
        
        # Set distance to self to infinity to exclude it
        dists[i] = np.inf
        
        # Find minimum distance to any other antenna
        min_dist = np.min(dists)
        if min_dist != np.inf:
            distances.append(min_dist)
    
    return distances

def calculate_operator_distances(data):
    """Calculate average minimum distances between antennas for each operator."""
    stats = {}
    
    # Group data by operator
    grouped = data.groupby('Exploitant')
    
    # Process each operator
    for operator, operator_data in tqdm(grouped, desc="Processing operators"):
        if len(operator_data) < 2:
            print(f"Skipping {operator} - insufficient data")
            continue
        
        # Process all antennas for this operator
        distances = process_operator_chunk((operator_data, len(operator_data)))
        
        if distances:
            stats[operator] = {
                'mean': np.mean(distances),
                'median': np.median(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'count': len(distances)
            }
    
    return stats

def main():
    # Load data
    print("Loading and merging data...")
    merged_data = load_data()
    
    # Validate data
    validate_coordinates(merged_data)
    
    # Calculate distances
    print("\nCalculating minimum distances between antennas for each operator...")
    results = calculate_operator_distances(merged_data)
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 100)
    print(f"{'Operator':<30} | {'Mean (km)':>10} | {'Median (km)':>10} | {'Std (km)':>10} | {'Min (km)':>10} | {'Max (km)':>10} | {'Count':>8}")
    print("-" * 100)
    
    for operator, stats in sorted(results.items(), key=lambda x: x[1]['mean']):
        print(f"{operator:<30} | {stats['mean']:10.2f} | {stats['median']:10.2f} | {stats['std']:10.2f} | {stats['min']:10.2f} | {stats['max']:10.2f} | {stats['count']:8d}")

if __name__ == "__main__":
    main() 