import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from itertools import combinations
from collections import defaultdict

def load_data(antennas_path='data/antennas.csv', locations_path='data/locations.csv'):
    """Load and merge antenna data with locations."""
    # Load antennas file
    antennas = pd.read_csv(antennas_path, delimiter=';', encoding='latin1')
    
    # Load locations file
    locations = pd.read_csv(locations_path, delimiter=';', encoding='latin1')
    
    # Rename column for merging
    locations = locations.rename(columns={'Numéro du support': 'Numéro de support'})
    
    # Merge datasets
    merged = pd.merge(antennas, locations, on='Numéro de support')
    
    # Select relevant columns
    merged = merged[['Numéro de support', 'Exploitant', 'Longitude', 'Latitude']]
    
    # Convert coordinates to numeric and clean data
    merged['Longitude'] = pd.to_numeric(merged['Longitude'], errors='coerce')
    merged['Latitude'] = pd.to_numeric(merged['Latitude'], errors='coerce')
    merged = merged.dropna(subset=['Longitude', 'Latitude'])
    
    return merged

def process_operator_chunk(args):
    """Process a chunk of antennas for a specific operator."""
    operator_data, chunk_size = args
    distances = []
    
    # Convert to numpy array for faster operations
    coordinates = operator_data[['Latitude', 'Longitude']].values
    
    for i in range(len(coordinates)):
        if i % chunk_size == 0:  # Process in chunks
            chunk_end = min(i + chunk_size, len(coordinates))
            current_coord = coordinates[i]
            
            # Calculate distances to all other points in the operator's dataset
            other_coords = coordinates[i+1:chunk_end]
            if len(other_coords) > 0:
                # Use numpy broadcasting for faster distance calculation
                lat1, lon1 = current_coord
                lat2, lon2 = other_coords[:, 0], other_coords[:, 1]
                
                # Vectorized Haversine formula
                R = 6371  # Earth's radius in kilometers
                
                dlat = np.radians(lat2 - lat1)
                dlon = np.radians(lon2 - lon1)
                
                a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                distances_km = R * c
                
                if len(distances_km) > 0:
                    min_dist = np.min(distances_km)
                    distances.append(min_dist)
    
    return distances

def calculate_operator_distances(data, chunk_size=1000):
    """Calculate average minimum distances between antennas for each operator."""
    results = {}
    
    # Group data by operator
    grouped = data.groupby('Exploitant')
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for operator, operator_data in tqdm(grouped, desc="Processing operators"):
            if len(operator_data) < 2:
                continue
                
            # Prepare chunks for parallel processing
            chunks = [(operator_data, chunk_size)] * ((len(operator_data) + chunk_size - 1) // chunk_size)
            
            # Process chunks in parallel
            distances = []
            for chunk_distances in executor.map(process_operator_chunk, chunks):
                distances.extend(chunk_distances)
            
            if distances:
                avg_distance = np.mean(distances)
                results[operator] = avg_distance
    
    return results

def main():
    # Load data
    print("Loading and merging data...")
    merged_data = load_data()
    
    # Calculate distances
    print("\nCalculating average minimum distances between antennas for each operator...")
    results = calculate_operator_distances(merged_data)
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    print("Operator | Average Minimum Distance (km)")
    print("-" * 50)
    for operator, avg_distance in sorted(results.items(), key=lambda x: x[1]):
        print(f"{operator:<30} | {avg_distance:.2f}")

if __name__ == "__main__":
    main() 