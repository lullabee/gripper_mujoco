import mujoco
import numpy as np
import matplotlib.pyplot as plt
import glfw
import time
import argparse
import os  # Import os for directory operations
#from spirob_gripper import SpiralGripper
from spirob_gripper2 import SpiralGripper2
from spirob_simulation import GripperSimulation

def count_meshes_in_directory(directory):
    """Count the number of mesh files in the given directory."""
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

if __name__ == "__main__":
    # Define the directory containing mesh files
    mesh_directory = "meshes2"  # Update this path to your mesh folder

    # Count the number of mesh files
    default_n_sections = count_meshes_in_directory(mesh_directory)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the gripper simulation.')
    parser.add_argument('--n_sections', type=int, default=default_n_sections, help='Number of sections for the gripper')
    args = parser.parse_args()
    
    # Create gripper and simulation with the specified number of sections
    gripper = SpiralGripper2(n_sections=args.n_sections)
    sim = GripperSimulation(gripper)
    
    # Initialize and run simulation
    if sim.init_visualization():
        sim.setup_callbacks()
        sim.run() 