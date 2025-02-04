import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import sys

print("Starting script...")  # Debug print

def load_and_merge_data(antennas_path, locations_path):
    print(f"Loading data from {antennas_path} and {locations_path}")
    try:
        # Load antennas file (we know it uses semicolon delimiter and latin1 encoding)
        print("Loading antennas file...")
        antennas = pd.read_csv(antennas_path, delimiter=';', encoding='latin1')
        print(f"Successfully loaded antennas: {len(antennas)} rows")
        print("\nUnique exploitants in the dataset:")
        print(antennas['Exploitant'].unique())
        print("\nFirst few rows of antennas:")
        print(antennas.head())
        print(f"\nAntennas columns: {antennas.columns.tolist()}")
        
        # Load locations file (we know it uses semicolon delimiter and latin1 encoding)
        print("\nLoading locations file...")
        locations = pd.read_csv(locations_path, delimiter=';', encoding='latin1')
        print(f"Successfully loaded locations: {len(locations)} rows")
        print("First few rows of locations:")
        print(locations.head())
        print(f"\nLocations columns: {locations.columns.tolist()}")
        
        # Merge on the common field
        print("\nMerging datasets...")
        
        # Rename the column in locations to match antennas
        locations = locations.rename(columns={'Numéro du support': 'Numéro de support'})
        
        # Merge the dataframes
        merged = pd.merge(antennas, locations, on='Numéro de support')
        
        # Select and reorder the columns we care about
        merged = merged[['Numéro de support', 'Exploitant', 'Longitude', 'Latitude']]
        
        # Optionally, ensure Longitude and Latitude are numeric:
        merged['Longitude'] = pd.to_numeric(merged['Longitude'], errors='coerce')
        merged['Latitude'] = pd.to_numeric(merged['Latitude'], errors='coerce')
        
        # Drop rows with missing coordinates
        merged = merged.dropna(subset=['Longitude', 'Latitude'])
        
        print(f"Successfully merged data: {len(merged)} rows")
        print("First few rows of merged data:")
        print(merged.head())
        return merged
            
    except Exception as e:
        print("Error reading or merging CSV files:", e)
        sys.exit(1)

def get_coordinates_from_address(address):
    """
    Use the Nominatim geocoder to convert an address to latitude and longitude.
    
    address: The parcel address as a string.
    
    Returns:
        tuple: (latitude, longitude)
    """
    geolocator = Nominatim(user_agent="antenna_locator")
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not geocode address: {address}")

def calculate_distance(coord1, coord2):
    """
    Calculate the geodesic distance between two (latitude, longitude) tuples.
    
    coord1: Tuple (lat, lon) for the first location.
    coord2: Tuple (lat, lon) for the second location.
    
    Returns:
        float: Distance in kilometers.
    """
    return geodesic(coord1, coord2).kilometers

def normalize_exploitant(name):
    """
    Normalize exploitant names to match the database format.
    """
    # Convert to uppercase for consistent matching
    name = name.upper()
    
    # Define common variations
    mappings = {
        'FREE': 'FREE MOBILE',
        'ORANGE FRANCE': 'ORANGE',
        'BOUYGUES': 'BOUYGUES TELECOM'
    }
    
    return mappings.get(name, name)

def find_closest_antenna(parcel_coords, merged_data, target_exploitant):
    """
    For a given parcel location and target exploitant, find the closest antenna.
    
    parcel_coords: Tuple (lat, lon) for the parcel.
    merged_data: The DataFrame containing antenna data.
    target_exploitant: String indicating which exploitant's antennas to consider.
    
    Returns:
        tuple: (closest_antenna_id, min_distance)
            - closest_antenna_id: 'Numéro de support' of the closest antenna
            - min_distance: Distance (in km) from the parcel.
    """
    # Normalize the target exploitant name
    normalized_exploitant = normalize_exploitant(target_exploitant)
    
    # Filter data to include only the antennas for the target exploitant
    subset = merged_data[merged_data['Exploitant'] == normalized_exploitant]
    
    if subset.empty:
        print(f"\nAvailable exploitants: {', '.join(sorted(merged_data['Exploitant'].unique()))}")
        return None, None

    min_distance = float('inf')
    closest_antenna = None
    for _, row in subset.iterrows():
        antenna_coords = (row['Latitude'], row['Longitude'])
        distance = calculate_distance(parcel_coords, antenna_coords)
        if distance < min_distance:
            min_distance = distance
            closest_antenna = row['Numéro de support']
    
    return closest_antenna, min_distance

def main():
    print("Entering main function...")
    # Paths to the CSV files
    antennas_csv = 'data/antennas.csv'
    locations_csv = 'data/locations.csv'
    
    # Load and merge data
    print("About to load and merge data...")
    merged_data = load_and_merge_data(antennas_csv, locations_csv)
    print("Data loaded and merged successfully")
    
    # Prompt the user for a parcel address
    print("\nEnter parcel address (in French):")
    address = input().strip()
    
    try:
        parcel_coords = get_coordinates_from_address(address)
        print(f"Coordinates for parcel: {parcel_coords}")
    except ValueError as e:
        print(e)
        sys.exit(1)
    
    # Ask the user to specify the exploitant
    print("\nEnter target exploitant:")
    target_exploitant = input().strip()
    
    # Find the closest antenna for the specified exploitant
    closest, distance = find_closest_antenna(parcel_coords, merged_data, target_exploitant)
    if closest is None:
        print(f"\nNo antennas found for exploitant: {target_exploitant}")
    else:
        print(f"\nClosest antenna (Numéro de support): {closest}")
        print(f"Distance to parcel: {distance:.2f} km")

if __name__ == "__main__":
    main()
