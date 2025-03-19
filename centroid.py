import pymeshlab
import os
import sys
import numpy as np

def recenter_stl(input_file, output_file):
    # Create a new MeshLab project
    ms = pymeshlab.MeshSet()
    
    # Load the STL file
    ms.load_new_mesh(input_file)

    # Get mesh information
    current_mesh = ms.current_mesh()
    
    # Get the bounding box and calculate center
    bbox_min = current_mesh.bounding_box().min()
    bbox_max = current_mesh.bounding_box().max()
    centroid = (bbox_min + bbox_max) / 2
    cx, cy, cz = centroid
    
    # Get vertex matrix (direct access to vertices)
    vertex_matrix = current_mesh.vertex_matrix()
    
    # Translate all vertices
    for i in range(vertex_matrix.shape[0]):
        vertex_matrix[i, 0] -= cx  # X coordinate
        vertex_matrix[i, 1] -= cy  # Y coordinate
        vertex_matrix[i, 2] -= cz  # Z coordinate
    
    # Create a new mesh with the transformed vertices
    new_mesh = pymeshlab.Mesh(vertex_matrix, current_mesh.face_matrix())
    
    # Clear the MeshSet and add the new mesh
    ms.clear()
    ms.add_mesh(new_mesh)
    
    # Save the recentered mesh
    ms.save_current_mesh(output_file)

    print(f"Recentered STL saved as: {output_file}")
    return cx, cy, cz

def process_folder(input_folder, output_folder=None):
    """Process all STL files in a folder and save recentered versions to output folder."""
    # If no output folder specified, create a 'recentered' subfolder in the input folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'recentered')
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Get all STL files in the input folder
    stl_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.stl')]
    
    if not stl_files:
        print(f"No STL files found in {input_folder}")
        return
    
    print(f"Found {len(stl_files)} STL files to process.")
    
    # Store centroid data to write to CSV later
    centroids = []
    
    # Process each STL file
    for stl_file in stl_files:
        input_path = os.path.join(input_folder, stl_file)
        output_path = os.path.join(output_folder, stl_file)
        
        print(f"Processing: {stl_file}")
        try:
            cx, cy, cz = recenter_stl(input_path, output_path)
            centroids.append((stl_file, cx, cy, cz))
        except Exception as e:
            print(f"Error processing {stl_file}: {e}")
            print(f"Try running with --debug to see more information")
            continue
    
    # Write centroids to CSV file
    if centroids:
        csv_path = os.path.join(output_folder, 'centroids.csv')
        with open(csv_path, 'w') as f:
            f.write("File,Centroid X,Centroid Y,Centroid Z\n")
            for filename, cx, cy, cz in centroids:
                f.write(f"{filename},{cx},{cy},{cz}\n")
        
        print(f"Processed {len(centroids)} files successfully.")
        print(f"Centroid data saved to {csv_path}")
    else:
        print("No files were processed successfully.")

def debug_mesh_info(mesh_file):
    """Print debugging information about a mesh file."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file)
    mesh = ms.current_mesh()
    
    print(f"\nDebug information for: {mesh_file}")
    print(f"Available methods in current_mesh:")
    for method in dir(mesh):
        if not method.startswith("_"):
            print(f"  - {method}")
    
    print("\nMesh information:")
    print(f"  - Number of vertices: {mesh.vertex_number()}")
    print(f"  - Number of faces: {mesh.face_number()}")
    
    # Try to get bounding box
    try:
        bbox = mesh.bounding_box()
        print(f"  - Bounding box min: {bbox.min()}")
        print(f"  - Bounding box max: {bbox.max()}")
    except Exception as e:
        print(f"  - Error getting bounding box: {e}")
    
    # List available methods in MeshSet
    print("\nAvailable methods in MeshSet:")
    for method in dir(ms):
        if not method.startswith("_"):
            print(f"  - {method}")

if __name__ == "__main__":
    # Check if folder path is provided as command line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list-filters":
            try:
                ms = pymeshlab.MeshSet()
                print("Available filters in pymeshlab:")
                for filter_name in ms.filter_list():
                    print(f"  - {filter_name}")
            except Exception as e:
                print(f"Error listing filters: {e}")
        elif sys.argv[1] == "--debug" and len(sys.argv) > 2:
            debug_mesh_info(sys.argv[2])
        else:
            input_folder = sys.argv[1]
            output_folder = sys.argv[2] if len(sys.argv) > 2 else None
            process_folder(input_folder, output_folder)
    else:
        # Example usage
        input_folder = "meshes2"  # Change this to your input folder
        output_folder = "meshes2_recentered"  # Change this to your desired output folder
        process_folder(input_folder, output_folder)