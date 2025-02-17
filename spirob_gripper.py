import mujoco
import numpy as np

class SpiralGripper:
    def __init__(self, n_sections=20, spiral_growth=0.4, taper_factor=0.7, mount_height=0.6):
        """
        Initialize a spiral-shaped gripper with elastic joints.
        n_sections: Number of segments per finger.
        spiral_growth: Growth factor of the logarithmic spiral.
        taper_factor: Reduction in width along the spiral.
        mount_height: Mount height of the gripper base.
        """
        self.n_sections = n_sections
        # self.spiral_growth = spiral_growth
        self.taper_factor = taper_factor
        self.mount_height = mount_height
        self.link_data = self.generate_finger_links(self.n_sections, 0.024, 0.048, 0.015, 0.015, 0.7)
        self.xml = self.generate_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        
        self.reset_gripper()

    def generate_finger_links(self, n_segments, a, b, initial_width, initial_length, taper_factor):
      """ Generate parameters for finger segments based on a logarithmic spiral. """
      segments = []
      theta_increment = 2 * np.pi / n_segments  # Uniform angle division
  
      for i in range(n_segments):
          theta = i * theta_increment
          r = a * np.exp(b * theta)
          width = initial_width * (taper_factor ** i)
          length = initial_length * (taper_factor ** i)
  
          segment = {
              'index': i,
              'theta': theta,
              'radius': r,
              'width': width,
              'length': length
          }
          segments.append(segment)
  
      return segments


    def get_finger_state(self, finger_idx):
        """Get the state of a finger (bend angles, etc.)."""
        # Calculate base index for this finger's joints
        base_idx = finger_idx * self.n_sections
        
        # Get joint angles for this finger
        joint_angles = []
        for i in range(self.n_sections):
            joint_id = base_idx + i
            angle = np.degrees(self.data.qpos[joint_id])
            joint_angles.append(angle)
        
        # Calculate average and max bend
        avg_bend = np.mean(joint_angles) if joint_angles else 0
        max_bend = np.max(np.abs(joint_angles)) if joint_angles else 0
        
        return {
            'avg_bend': avg_bend,
            'max_bend': max_bend,
            'joint_angles': joint_angles
        } 

    def get_contact_info(self):
        """Get information about current contacts."""
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            contact_info = {
                'position': contact.pos.copy(),
                'force': contact.frame[0]  # Assuming frame[0] holds the contact force magnitude
            }
            contacts.append(contact_info)
        return contacts

    def generate_finger_xml(self, name, base_x, base_y, base_angle):
        """Create MuJoCo XML for a finger with vertically-aligned segments."""
        xml = f'<body name="{name}_base" pos="{base_x} {base_y} 0" euler="0 0 {base_angle}">'
        
        for i, link in enumerate(self.link_data):
            xml += f'''
            <body name="{name}_section_{i}" pos="0 0 {-0.007 - i*0.007}">
                <joint name="{name}_joint_{i}" type="hinge" axis="1 0 0" range="-90 90" stiffness="0.01" damping="0.1"/>
                <geom type="mesh" mesh="segment_{i}" 
                      euler="90 90 90"
                      rgba="0.4 0.4 0.8 1"/>
                <!-- Sites aligned with scaled mesh cylinders -->
                <site name="{name}_elastic_site_{i}" pos="0 0 0" size="0.0004" rgba="0.2 0.2 1 1"/>
                <site name="{name}_tendon_front_{i}" pos="0.02 0.02 0" size="0.0004" rgba="1 0 0 1"/>
                <site name="{name}_tendon_back_{i}" pos="0.02 -0.02 0" size="0.0004" rgba="1 0 0 1"/>
            </body>
            '''
        xml += '</body>'
        return xml

    def generate_elastic_xml(self, name):
        """Generate XML for elastic connections between segments."""
        xml = ""
        for i in range(self.n_sections - 1):
            xml += f'''
            <spatial name="{name}_elastic_{i}" width="0.002" rgba="0.2 0.2 1 1" stiffness="0.1" damping="0.01">
                <site site="{name}_elastic_site_{i}"/>
                <site site="{name}_elastic_site_{i+1}"/>
            </spatial>
            '''
        return xml

    def generate_tendon_xml(self, name):
        """Create actuation tendons for finger control."""
        # Front tendon
        xml = f'''
        <spatial name="{name}_tendon_front" width="0.0002" rgba="1 0 0 1" stiffness="1" damping="0.1">
            <site site="{name}_base"/>
        '''
        for i in range(self.n_sections):
            xml += f'<site site="{name}_tendon_front_{i}"/>'
        xml += '</spatial>'
        
        # Back tendon
        xml += f'''
        <spatial name="{name}_tendon_back" width="0.0002" rgba="1 0 0 1" stiffness="1" damping="0.1">
            <site site="{name}_base"/>
        '''
        for i in range(self.n_sections):
            xml += f'<site site="{name}_tendon_back_{i}"/>'
        xml += '</spatial>'
        return xml

    def generate_xml(self):
        xml = f'''
        <mujoco>
            <option gravity="0 0 -9.81"/>
            <asset>
                {self.generate_mesh_assets()}
            </asset>
            <default>
                <joint damping="0.1"/>
                <tendon width="0.001" stiffness="1" damping="0.1"/>
            </default>
            <worldbody>
                <body name="gripper_mount" pos="0 0 {self.mount_height}">
                    <joint name="mount_joint" type="hinge" axis="0 0 1" range="-0.0001 0.0001" damping="1000" stiffness="1000"/>
                    <geom type="box" size="0.05 0.05 0.01" rgba="0.5 0.5 0.5 0.3"/>
                    <body name="gripper_base" pos="0 0 -0.02">
                        <geom type="box" size="0.04 0.04 0.02" rgba="0.7 0.7 0.7 0.3"/>
                        <site name="finger1_base" pos="0.03 0 0" size="0.002" rgba="1 0 0 1"/>
                        <site name="finger2_base" pos="-0.03 0 0" size="0.002" rgba="1 0 0 1"/>
                        {self.generate_finger_xml("finger1", 0.03, 0, 45)}
                        {self.generate_finger_xml("finger2", -0.03, 0, -45)}
                    </body>
                </body>
                <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
            </worldbody>
            <tendon>
                {self.generate_elastic_xml("finger1")}
                {self.generate_elastic_xml("finger2")}
                {self.generate_tendon_xml("finger1")}
                {self.generate_tendon_xml("finger2")}
            </tendon>
            <actuator>
                <motor tendon="finger1_tendon_front" ctrlrange="-2 2" gear="10"/>
                <motor tendon="finger1_tendon_back" ctrlrange="-2 2" gear="10"/>
                <motor tendon="finger2_tendon_front" ctrlrange="-2 2" gear="10"/>
                <motor tendon="finger2_tendon_back" ctrlrange="-2 2" gear="10"/>
            </actuator>
        </mujoco>
        '''
        return xml
    
    def reset_gripper(self, force=0):
        """Reset gripper to neutral position."""
        self.data.ctrl[:] = force

    def close_gripper(self, max_force=0.25):
        """Close gripper by tightening front tendons and loosening back tendons."""
        self.data.ctrl[::2] = max_force  # Front tendons
        self.data.ctrl[1::2] = -max_force  # Back tendons

    def open_gripper(self, max_force=0.2):
        """Open gripper by tightening back tendons and loosening front tendons."""
        self.data.ctrl[::2] = -max_force  # Front tendons
        self.data.ctrl[1::2] = max_force  # Back tendons

    def generate_mesh_assets(self):
        """Generate XML for mesh assets."""
        assets = ""
        for i in range(self.n_sections):
            assets += f'<mesh name="segment_{i}" file="meshes/segment_{i}.stl" scale="2 2 2"/>\n'
        return assets
