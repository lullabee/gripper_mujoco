import os
import numpy as np
import csv
from stl import mesh

def calculate_stl_center(stl_file):
    """Calculate the center of an STL file."""
    try:
        # Load the STL file
        mesh_data = mesh.Mesh.from_file(stl_file)
        
        # Get all vertices
        vertices = np.vstack((
            mesh_data.v0, mesh_data.v1, mesh_data.v2
        ))
        
        # Calculate the center as the average of all vertices
        center = vertices.mean(axis=0)
        
        # Calculate the min and max bounds
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        
        # Calculate center from bounds (alternative method)
        center_from_bounds = (min_bounds + max_bounds) / 2
        
        return {
            'file': os.path.basename(stl_file),
            'center_avg': center,
            'center_bounds': center_from_bounds,
            'min_bounds': min_bounds,
            'max_bounds': max_bounds
        }
    except Exception as e:
        print(f"Error processing {stl_file}: {e}")
        return None

def main():
    # Set the directory containing STL files
    stl_dir = "meshes2"  # Change to your STL directory
    
    # Ensure the directory exists
    if not os.path.exists(stl_dir):
        print(f"Directory {stl_dir} does not exist.")
        return
    
    # Get all STL files in the directory
    stl_files = [os.path.join(stl_dir, f) for f in os.listdir(stl_dir) if f.lower().endswith('.stl')]
    
    if not stl_files:
        print(f"No STL files found in {stl_dir}.")
        return
    
    # Calculate centers for all STL files
    results = []
    for stl_file in stl_files:
        result = calculate_stl_center(stl_file)
        if result:
            results.append(result)
    
    # Sort results by filename
    results.sort(key=lambda x: x['file'])
    
    # Write results to CSV
    csv_file = "stl_centers.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['File', 'Center X (Avg)', 'Center Y (Avg)', 'Center Z (Avg)', 
                        'Center X (Bounds)', 'Center Y (Bounds)', 'Center Z (Bounds)',
                        'Min X', 'Min Y', 'Min Z', 'Max X', 'Max Y', 'Max Z'])
        
        # Write data
        for result in results:
            writer.writerow([
                result['file'],
                *result['center_avg'],
                *result['center_bounds'],
                *result['min_bounds'],
                *result['max_bounds']
            ])
    
    print(f"Centers extracted and saved to {csv_file}")
    
    # Generate offsets (negation of center) for easy copying to MuJoCo
    with open("stl_offsets.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['File', 'Offset X', 'Offset Y', 'Offset Z'])
        
        # Write data (negative of center to offset it)
        for result in results:
            center = result['center_bounds']  # Using bounds-based center
            writer.writerow([
                result['file'],
                -center[0],
                -center[1],
                -center[2]
            ])
    
    print(f"Offsets for centering STLs saved to stl_offsets.csv")

if __name__ == "__main__":
    main() 