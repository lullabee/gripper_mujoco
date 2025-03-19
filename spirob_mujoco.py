import numpy as np
import matplotlib.pyplot as plt
import argparse
import os  # Import os for directory operations
from spirob_simulation import GripperSimulation  # Import the GripperSimulation class

def count_meshes_in_directory(directory):
    """Count the number of mesh files in the given directory."""
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def parse_dimension_args():
    """
    Parse command line arguments to determine whether to use 2D or 3D mode.
    Returns a tuple of (suffix_string, meshes_folder_name)
    """
    parser = argparse.ArgumentParser(description='Spiral Gripper Simulation')
    parser.add_argument('--2d', dest='use_2d', action='store_true', 
                        help='Use 2D mode instead of 3D mode')
    parser.add_argument('--input-folder', type=str, default=None,
                        help='Input folder for mesh files')
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Output folder for recentered mesh files')
    
    args = parser.parse_args()
    
    # Determine the suffix based on the argument
    dim_suffix = "2d" if args.use_2d else "3d"
    
    # Create folder names
    meshes_folder = f"meshes_{dim_suffix}"
    output_folder = args.output_folder
    
    if output_folder is None and args.input_folder is not None:
        # If input folder specified but not output, create output as a subfolder
        output_folder = os.path.join(args.input_folder, f'recentered_{dim_suffix}')
    elif output_folder is None:
        # Default output folder if nothing specified
        output_folder = f"meshes_{dim_suffix}_recentered"
    
    # Get input folder
    input_folder = args.input_folder if args.input_folder else meshes_folder
    
    return {
        'suffix': dim_suffix,
        'input_folder': input_folder,
        'output_folder': output_folder,
        'use_2d': args.use_2d
    }

def import_gripper_class(dim_config):
    """
    Import the appropriate gripper class based on the dimension configuration.
    
    Args:
        dim_config: Dictionary containing configuration info from parse_dimension_args()
        
    Returns:
        The appropriate SpiralGripper class
    """
    if dim_config['use_2d']:
        from spirob_gripper_2d import SpiralGripper2D
        return SpiralGripper2D
    else:
        from spirob_gripper_3d import SpiralGripper3D
        return SpiralGripper3D

if __name__ == "__main__":
    dim_config = parse_dimension_args()

    # Define the directory containing mesh files
    mesh_directory = dim_config['input_folder']  # Use the input folder from dim_config

    # Count the number of mesh files
    default_n_sections = count_meshes_in_directory(mesh_directory)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the gripper simulation.')
    parser.add_argument('--n_sections', type=int, default=default_n_sections, help='Number of sections for the gripper')
    args = parser.parse_args()
    
    # Example of importing the right gripper class
    try:
        GripperClass = import_gripper_class(dim_config)
        print(f"Successfully imported {GripperClass.__name__}")
        
        # Create gripper instance with the correct mesh folder
        gripper = GripperClass(n_sections=args.n_sections, mesh_folder=mesh_directory)
        print(f"Using mesh folder: {mesh_directory}")
        
    except ImportError as e:
        print(f"Failed to import gripper class: {e}")
        print("Make sure the appropriate module exists (spirob_gripper_2d.py or spirob_gripper_3d.py)")
        exit(1)
    
    sim = GripperSimulation(gripper)
    # Initialize and run simulation
    if sim.init_visualization():  # Fixed method name typo
        sim.setup_callbacks()
        sim.run() 