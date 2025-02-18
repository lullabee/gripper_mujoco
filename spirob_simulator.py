import matplotlib
matplotlib.use('macosx')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from svg.path import parse_path
from xml.dom import minidom
import mujoco

class SpiRob:
    def __init__(self, scale=1.0, length=5.0, cross_section_type='triangle', n_sections=20, svg_path=None):
        """
        Initialize SpiRob finger parameters
        scale: overall size scaling
        length: length of the finger
        cross_section_type: shape of cross section ('triangle', 'square', or path to svg file)
        n_sections: number of discrete sections
        """
        self.scale = scale
        self.length = length
        self.cross_section_type = cross_section_type
        self.n_sections = n_sections
        self.b = 0.22  # Default taper angle parameter
        self.svg_points = self.load_svg(svg_path) if svg_path else None
        
    def load_svg(self, svg_path):
        """Load and parse SVG file to extract points"""
        doc = minidom.parse(svg_path)
        path_strings = [path.getAttribute('d') for path 
                       in doc.getElementsByTagName('path')]
        doc.unlink()
        
        if not path_strings:
            raise ValueError("No paths found in SVG file")
            
        # Parse the first path
        path = parse_path(path_strings[0])
        
        # Sample points along the path
        n_points = 32  # Number of points to sample
        points = []
        for i in range(n_points):
            t = i / (n_points - 1)
            point = path.point(t)
            points.append([point.real, point.imag])
            
        return np.array(points)
        
    def get_cross_section_points(self, center, radius, angle, cross_type, thickness=0.1):
        """Generate points for the cross-section with thickness and height"""
        if self.svg_points is not None:
            # Use loaded SVG points
            points = self.svg_points.copy()
            
            # Scale points to match radius
            points -= np.mean(points, axis=0)  # Center points
            max_dim = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
            points *= (2 * radius / max_dim)  # Scale to match radius
            
            # Rotate points
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            points = points @ rot_matrix
            
            # Translate to center
            points += np.array([center[0], center[1]])
            
            # Generate front and back faces
            front_points = np.column_stack([points, np.full(len(points), center[2] + thickness/2)])
            back_points = np.column_stack([points, np.full(len(points), center[2] - thickness/2)])
            
            return front_points, back_points
        else:
            if cross_type == 'triangle':
                n_points = 3
                offset = 0
                height_factor = 1.5  # Triangle is 1.5x taller than wide
            else:  # square
                n_points = 4
                offset = np.pi/4
                height_factor = 2.0  # Rectangle is 2x taller than wide
            
            # Generate front and back face points
            front_points = []
            back_points = []
            thickness = thickness * radius
            
            for i in range(n_points):
                theta = 2 * np.pi * i / n_points + offset + angle
                
                # Apply height factor to y-coordinate
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta) * height_factor
                
                # Front face (higher z)
                front_points.append([x, y, center[2] + thickness/2])
                # Back face (lower z)
                back_points.append([x, y, center[2] - thickness/2])
            
            return np.array(front_points), np.array(back_points)

    def generate_finger_mesh(self, curl_factor=0.0):
        """Generate 3D mesh points for the finger with spaces between sections"""
        section_length = self.length / self.n_sections
        base_radius = 0.2 * self.scale
        spacing = 0.3 * section_length  # Space between sections
        
        vertices = []
        faces = []
        vertex_count = 0
        
        # Generate curved centerline
        t = np.linspace(0, 1, self.n_sections)
        curvature = curl_factor * np.pi
        
        x_center = self.scale * (1 - np.cos(curvature * t))
        y_center = np.zeros_like(t)
        z_center = self.scale * (t * self.length - np.sin(curvature * t))
        
        for i in range(self.n_sections):
            # Calculate local orientation
            if i < self.n_sections - 1:
                dx = x_center[i+1] - x_center[i]
                dz = z_center[i+1] - z_center[i]
                angle = np.arctan2(dx, dz)
            else:
                angle = curvature
                
            # Calculate local radius with taper
            taper_factor = 1 - 0.5 * (i / (self.n_sections - 1))
            radius = base_radius * taper_factor
            
            # Generate front and back faces of the section
            center = [x_center[i], y_center[i], z_center[i]]
            front_points, back_points = self.get_cross_section_points(
                center, radius, angle, self.cross_section_type)
            
            # Add all points to vertices
            section_vertices = np.vstack((front_points, back_points))
            vertices.extend(section_vertices)
            
            # Modify face generation for SVG case
            if self.svg_points is not None:
                n_points = len(self.svg_points)
                
                # Front face (use triangulation)
                from scipy.spatial import Delaunay
                front_points_2d = front_points[:, :2]
                tri = Delaunay(front_points_2d)
                for simplex in tri.simplices:
                    faces.append([vertex_count + s for s in simplex])
                    
                # Back face (mirror of front)
                for simplex in tri.simplices:
                    faces.append([vertex_count + n_points + s for s in simplex])
                    
                # Side faces
                for i in range(n_points):
                    i_next = (i + 1) % n_points
                    faces.extend([
                        [vertex_count + i, 
                         vertex_count + i_next,
                         vertex_count + n_points + i],
                        [vertex_count + i_next,
                         vertex_count + n_points + i_next,
                         vertex_count + n_points + i]
                    ])
            else:
                # Generate faces for the section
                n_points = 3 if self.cross_section_type == 'triangle' else 4
                
                # Front face
                for j in range(n_points):
                    faces.append([
                        vertex_count + j,
                        vertex_count + (j+1)%n_points,
                        vertex_count + (j+2)%n_points
                    ])
                
                # Back face
                for j in range(n_points):
                    faces.append([
                        vertex_count + n_points + j,
                        vertex_count + n_points + (j+1)%n_points,
                        vertex_count + n_points + (j+2)%n_points
                    ])
                
                # Side faces
                for j in range(n_points):
                    j_next = (j + 1) % n_points
                    faces.append([
                        vertex_count + j,
                        vertex_count + j_next,
                        vertex_count + n_points + j
                    ])
                    faces.append([
                        vertex_count + j_next,
                        vertex_count + n_points + j_next,
                        vertex_count + n_points + j
                    ])
            
            vertex_count += 2 * n_points
            
        return np.array(vertices), np.array(faces)

    def generate_gripper_mesh(self, curl_factors=(0.0, 0.0, 0.0)):
        """Generate 3D mesh for three-fingered gripper"""
        vertices_list = []
        faces_list = []
        vertex_count = 0
        
        # Finger positions (120 degrees apart)
        angles = [0, 2*np.pi/3, 4*np.pi/3]
        base_offset = self.scale * 0.5  # Distance from center
        
        for i, angle in enumerate(angles):
            # Calculate finger base position
            base_x = base_offset * np.cos(angle)
            base_y = base_offset * np.sin(angle)
            
            # Generate finger mesh
            vertices, faces = self.generate_finger_mesh(curl_factors[i])
            
            # Rotate and translate vertices for this finger
            rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            # Apply rotation and translation
            vertices = vertices @ rot_matrix
            vertices[:, 0] += base_x
            vertices[:, 1] += base_y
            
            # Add vertices and update faces
            vertices_list.append(vertices)
            faces_list.append(faces + vertex_count)
            vertex_count += len(vertices)
        
        # Combine all vertices and faces
        vertices = np.vstack(vertices_list)
        faces = np.vstack(faces_list)
        
        return vertices, faces

    def interactive_demo(self):
        """Create interactive 3D visualization for three-fingered gripper"""
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 4)
        ax_3d = fig.add_subplot(gs[:, :-1], projection='3d')
        
        # Initial parameters
        vertices, faces = self.generate_gripper_mesh()
        
        # Plot the 3D surface
        mesh = ax_3d.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                 triangles=faces, cmap='viridis', alpha=0.8)
        
        # Setup 3D plot
        ax_3d.set_xlim(-self.scale*3, self.scale*3)
        ax_3d.set_ylim(-self.scale*3, self.scale*3)
        ax_3d.set_zlim(0, self.length)
        ax_3d.set_title('SpiRob Gripper Configuration')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        
        # Add sliders
        ax_load = plt.axes([0.75, 0.85, 0.15, 0.05])
        ax_scale = plt.axes([0.75, 0.75, 0.15, 0.02])
        ax_length = plt.axes([0.75, 0.65, 0.15, 0.02])
        ax_sections = plt.axes([0.75, 0.55, 0.15, 0.02])
        ax_curl = plt.axes([0.75, 0.45, 0.15, 0.02])
        ax_curl_diff = plt.axes([0.75, 0.35, 0.15, 0.02])
        ax_type = plt.axes([0.75, 0.2, 0.15, 0.1])
        
        s_scale = Slider(ax_scale, 'Scale', 0.5, 2.0, valinit=self.scale)
        s_length = Slider(ax_length, 'Length', 3.0, 10.0, valinit=self.length)
        s_sections = Slider(ax_sections, 'Sections', 5, 30, valinit=self.n_sections, valstep=1)
        s_curl = Slider(ax_curl, 'Curl', 0.0, 1.0, valinit=0.0)
        s_curl_diff = Slider(ax_curl_diff, 'Curl Diff', 0.0, 0.5, valinit=0.0)
        
        button_load = Button(ax_load, 'Load SVG', color='lightblue')
        radio = RadioButtons(ax_type, ['triangle', 'square'], active=0)
        
        def update(val=None):
            """Update visualization when parameters change"""
            self.scale = s_scale.val
            self.length = s_length.val
            self.n_sections = int(s_sections.val)
            
            # Calculate individual finger curls
            base_curl = s_curl.val
            curl_diff = s_curl_diff.val
            curls = (
                base_curl,
                base_curl + curl_diff,
                base_curl + curl_diff
            )
            
            # Generate new mesh
            vertices, faces = self.generate_gripper_mesh(curls)
            
            # Update plot
            ax_3d.clear()
            ax_3d.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                             triangles=faces, cmap='viridis', alpha=0.8,
                             shade=True, lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=45))
            
            # Maintain view settings
            ax_3d.set_xlim(-self.scale*3, self.scale*3)
            ax_3d.set_ylim(-self.scale*3, self.scale*3)
            ax_3d.set_zlim(0, self.length)
            ax_3d.set_title('SpiRob Gripper Configuration')
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            
            fig.canvas.draw_idle()
        
        def update_type(label):
            """Update cross-section type"""
            self.cross_section_type = label
            update()
        
        def load_svg_file(event):
            """Handle SVG file loading"""
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()  # Hide the main window
            
            file_path = filedialog.askopenfilename(
                title='Select SVG file',
                filetypes=[('SVG files', '*.svg')]
            )
            
            if file_path:
                try:
                    self.svg_points = self.load_svg(file_path)
                    update()
                except Exception as e:
                    print(f"Error loading SVG: {e}")
            
            root.destroy()
        
        # Connect callbacks
        s_scale.on_changed(update)
        s_length.on_changed(update)
        s_sections.on_changed(update)
        s_curl.on_changed(update)
        s_curl_diff.on_changed(update)
        button_load.on_clicked(load_svg_file)
        radio.on_clicked(update_type)
        
        plt.show()

class SpirobSimulator:
    def __init__(self, n_sections=10):
        self.gripper = SpiralGripper(n_sections=n_sections)
        self.viewer = None
        self.add_objects_to_scene()
        
    def add_objects_to_scene(self):
        """Add objects to the scene by modifying the XML."""
        # Parse existing XML
        xml = self.gripper.xml
        
        # Add objects before the ground plane
        objects_xml = '''
            <!-- Objects to grasp -->
            <body name="sphere1" pos="0 0 0.1">
                <joint type="free"/>
                <geom type="sphere" size="0.02" rgba="1 0.5 0.5 1" mass="0.1"/>
            </body>
            <body name="sphere2" pos="0.1 0 0.1">
                <joint type="free"/>
                <geom type="sphere" size="0.015" rgba="0.5 1 0.5 1" mass="0.1"/>
            </body>
            <body name="box1" pos="-0.1 0 0.1">
                <joint type="free"/>
                <geom type="box" size="0.02 0.02 0.02" rgba="0.5 0.5 1 1" mass="0.1"/>
            </body>
        '''
        
        # Insert objects before ground plane
        xml = xml.replace('<geom type="plane"', objects_xml + '\n<geom type="plane"')
        
        # Update model with new XML
        self.gripper.xml = xml
        self.gripper.model = mujoco.MjModel.from_xml_string(xml)
        self.gripper.data = mujoco.MjData(self.gripper.model)
        
        # Reset gripper after modifying scene
        self.gripper.reset_gripper()

    def simulate(self, duration=10.0, render=True):
        """Run simulation for specified duration."""
        if render and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.gripper.model, self.gripper.data)

        # Simulation loop
        start_time = self.gripper.data.time
        while self.gripper.data.time - start_time < duration:
            mujoco.mj_step(self.gripper.model, self.gripper.data)
            
            if render:
                self.viewer.sync()
                
            # Add keyboard controls
            if render and self.viewer.is_running():
                key_events = self.viewer.input.key_events
                for key in key_events:
                    if key.key == '1':  # Close gripper
                        self.gripper.close_gripper()
                    elif key.key == '2':  # Open gripper
                        self.gripper.open_gripper()
                    elif key.key == '3':  # Reset gripper
                        self.gripper.reset_gripper()
                key_events.clear()
            else:
                break

# Example usage
if __name__ == "__main__":
    print("Running SpirobSimulator...")
    print("Controls:")
    print("1: Close gripper")
    print("2: Open gripper")
    print("3: Reset gripper")
    
    simulator = SpirobSimulator(n_sections=10)
    simulator.simulate() 