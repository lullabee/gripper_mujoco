import numpy as np
import matplotlib.pyplot as plt
from stl.mesh import Mesh
import trimesh
import os
import glob

def spiral(a, b, theta):
    """Calculate points for spiral segment."""
    r = a * np.exp(b * theta)
    rc = (a * np.exp(b * theta) + a * np.exp(b * (theta + 2 * np.pi))) / 2
    
    x1 = np.cos(theta) * r
    y1 = np.sin(theta) * r
    x2 = np.cos(theta) * rc
    y2 = np.sin(theta) * rc
    return x1, y1, x2, y2

def generate_segment_mesh(a, b, theta, scale, thickness, taper_min, taper_rate):
    """Generate a mesh for a single trapezoid-shaped finger segment."""
    # Get points for front face
    x1, y1, x2, y2 = spiral(a, b, theta)
    x3, y3, x4, y4 = spiral(a, b, theta + np.deg2rad(15))
    
    # Calculate taper factor based on position in spiral
    taper = 1.0 - (taper_rate * theta / (2 * np.pi))
    thickness = thickness * max(taper_min, taper)
    
    # Create vertices for main segment
    vertices = np.zeros((8, 3))
    # Front face (smaller end)
    vertices[0] = [x1 * scale, y1 * scale, -thickness/2]
    vertices[1] = [x2 * scale, y2 * scale, -thickness/2]
    vertices[2] = [x3 * scale, y3 * scale, -thickness/2]
    vertices[3] = [x4 * scale, y4 * scale, -thickness/2]
    # Back face
    vertices[4:8] = vertices[0:4] + np.array([0, 0, thickness])
    
    # Also taper the width of the segment
    width_scale = max(0.3, taper)
    center = np.mean(vertices, axis=0)
    vertices = center + (vertices - center) * np.array([width_scale, width_scale, 1])
    
    # Define faces for main segment
    faces = np.array([
        # Front face
        [0, 2, 1], [1, 2, 3],
        # Back face
        [4, 5, 6], [5, 7, 6],
        # Side faces
        [0, 1, 4], [1, 5, 4],  # Bottom
        [1, 3, 7], [1, 7, 5],  # Right
        [0, 4, 2], [2, 4, 6],  # Left
        [2, 6, 3], [3, 6, 7]   # Top
    ])
    
    # Get the outer edge points (will be our mirror line)
    edge_start = vertices[1]  # Point 2 (outer edge start)
    edge_end = vertices[3]    # Point 4 (outer edge end)
    
    # Find center of mirror line
    mirror_center = (edge_start + edge_end) / 2
    mirror_center[2] = 0  # Keep Z unchanged
    
    # Center vertices on mirror line
    vertices = vertices - mirror_center
    
    # Calculate mirror transform
    mirror_vec = edge_end - edge_start
    mirror_length = np.sqrt(np.sum(mirror_vec**2))
    mirror_dir = mirror_vec / mirror_length
    mirror_normal = np.array([-mirror_dir[1], mirror_dir[0], 0])
    
    def mirror_point(p):
        # Project onto mirror normal and reflect
        proj = 2 * np.dot(p, mirror_normal)
        return p - proj * mirror_normal
    
    # Mirror vertices
    mirrored_vertices = np.zeros_like(vertices)
    for i in range(len(vertices)):
        mirrored_vertices[i] = mirror_point(vertices[i])
    
    # Combine meshes
    combined_vertices = np.vstack([vertices, mirrored_vertices])
    
    # Create mirrored faces with reversed winding order
    mirrored_faces = faces.copy()
    for i in range(len(faces)):
        mirrored_faces[i] = [faces[i][0], faces[i][2], faces[i][1]]
    mirrored_faces += len(vertices)
    
    combined_faces = np.vstack([faces, mirrored_faces])
    
    # Create final mesh
    segment = Mesh(np.zeros(combined_faces.shape[0], dtype=Mesh.dtype))
    for i, f in enumerate(combined_faces):
        for j in range(3):
            segment.vectors[i][j] = combined_vertices[f[j],:]
    
    return segment

def generate_finger_segments(n_segments, a, b, scale, thickness, taper_min, 
                           taper_rate, angle_increment):
    """Generate and save meshes for all finger segments."""
    if not os.path.exists('meshes'):
        os.makedirs('meshes')
    
    for i in range(n_segments):
        theta = i * angle_increment
        segment = generate_segment_mesh(
            a, b, theta, scale, thickness, taper_min, taper_rate)
        segment.save(f'meshes/segment_{i}.stl')

def preview_segments(a, b, n_segments, angle_increment=np.deg2rad(15)):
    """Preview the spiral segments in 2D with labeled mirror face."""
    plt.figure(figsize=(10, 10))
    
    # Plot each segment
    for i in range(n_segments):
        theta = i * angle_increment
        x1, y1, x2, y2 = spiral(a, b, theta)
        x3, y3, x4, y4 = spiral(a, b, theta + angle_increment)
        
        # Plot original segment outline
        plt.plot([x1, x2, x4, x3, x1], [y1, y2, y4, y3, y1], 'b-')
        plt.text(x1, y1, str(i), fontsize=8)
        
        # Highlight mirror face
        plt.plot([x2, x4], [y2, y4], 'r-', linewidth=2)
        
        # Calculate mirror line direction vector
        mirror_dx = x4 - x2
        mirror_dy = y4 - y2
        mirror_length = np.sqrt(mirror_dx**2 + mirror_dy**2)
        mirror_nx = -mirror_dy/mirror_length  # Normal vector x
        mirror_ny = mirror_dx/mirror_length   # Normal vector y
        
        # Mirror points across the red line
        def mirror_point(px, py):
            # Vector from point to mirror line start
            vx = px - x2
            vy = py - y2
            # Project onto mirror line normal
            proj = 2*(vx*mirror_nx + vy*mirror_ny)
            # Mirror point
            return px - proj*mirror_nx, py - proj*mirror_ny
        
        # Mirror all points
        mirror_x1, mirror_y1 = mirror_point(x1, y1)
        mirror_x2, mirror_y2 = mirror_point(x2, y2)
        mirror_x3, mirror_y3 = mirror_point(x3, y3)
        mirror_x4, mirror_y4 = mirror_point(x4, y4)
        
        # Plot mirrored outline
        plt.plot([mirror_x1, mirror_x2, mirror_x4, mirror_x3, mirror_x1], 
                 [mirror_y1, mirror_y2, mirror_y4, mirror_y3, mirror_y1], 
                 'b--', alpha=0.5)
        
        if i == 0:  # Only label on first segment for clarity
            mid_x = (x2 + x4) / 2
            mid_y = (y2 + y4) / 2
            plt.text(mid_x, mid_y, 'Mirror Face\n(Outer Edge)', 
                    color='red', fontsize=10, 
                    horizontalalignment='center',
                    verticalalignment='center')
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('Spiral Segments Preview\nRed line shows mirror face, dashed lines show mirrored result')
    plt.show()

def preview_stl(segment_number):
    """Preview a specific STL file using trimesh."""
    # Load the mesh
    mesh = trimesh.load(f'meshes/segment_{segment_number}.stl')
    
    # Show the mesh
    mesh.show()

def clean_mesh_folder():
    """Remove all files in the meshes directory."""
    files = glob.glob('meshes/*')
    for f in files:
        os.remove(f)

if __name__ == "__main__":
    # Configuration parameters
    params = {
        'n_segments': 10,
        'a': 0.01,          # Initial radius
        'b': 0.32,          # Growth rate
        'scale': 0.5,       # Overall scale factor
        'thickness': 0.01,  # Base thickness
        'taper_min': 0.3,   # Minimum taper (don't go below 30%)
        'taper_rate': 0.9,  # How quickly segments get thinner
        'angle_increment': np.deg2rad(15),  # Angle between segments
        'attachment_radius': 0.0002,  # Radius of attachment points
        'attachment_height': 0.0003,  # Height of attachment points
    }
    
    clean_mesh_folder()
    preview_segments(params['a'], params['b'], params['n_segments'])
    generate_finger_segments(
        n_segments=params['n_segments'],
        a=params['a'],
        b=params['b'],
        scale=params['scale'],
        thickness=params['thickness'],
        taper_min=params['taper_min'],
        taper_rate=params['taper_rate'],
        angle_increment=params['angle_increment']
    )
    preview_stl(0)