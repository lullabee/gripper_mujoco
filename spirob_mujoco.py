import mujoco
import numpy as np
import matplotlib.pyplot as plt
import glfw
import time

class SpiRobGripper:
    def __init__(self, n_sections=20, mount_height=0.3):
        """
        Initialize SpiRob gripper
        n_sections: number of sections per finger
        mount_height: height to mount the gripper base from ground
        """
        self.mount_height = mount_height
        self.xml = self.generate_xml(n_sections)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
    def generate_finger_xml(self, name, x, y, angle, n_sections):
        # Parameters from the paper
        section_length = 0.015
        base_width = 0.012
        base_height = 0.020
        taper_factor = 0.7
        spacing = 0.001
        
        # Start building the finger XML
        xml = f"""
            <body name="{name}_base" pos="{x} {y} 0" euler="0 0 {angle}">
                <site name="{name}_flexor_base" pos="{base_width/2} 0 0" size="0.002" rgba="1 0 0 0.5"/>
                <site name="{name}_extensor_base" pos="{-base_width/2} 0 0" size="0.002" rgba="0 1 0 0.5"/>
        """
        
        # Create chain of sections
        current_body = ""
        for i in range(n_sections):
            scale = 1 - (i/n_sections) * (1-taper_factor)
            current_width = base_width * scale
            current_height = base_height * scale
            
            current_body += f"""
                <body name="{name}_section_{i}" pos="0 0 {section_length + spacing}">
                    <joint name="{name}_joint_{i}" 
                           type="hinge" 
                           axis="1 0 0" 
                           range="-45 45"
                           damping="0.01"
                           stiffness="0.001"
                           armature="0.001"/>
                    
                    <!-- Triangular prism using cylinders -->
                    <geom type="cylinder" 
                          fromto="{-current_width/2} {-current_height/2} 0  {current_width/2} {-current_height/2} 0"
                          size="0.002"
                          rgba=".4 .4 .8 1"/>
                    <geom type="cylinder" 
                          fromto="{current_width/2} {-current_height/2} 0  0 {current_height/2} 0"
                          size="0.002"
                          rgba=".4 .4 .8 1"/>
                    <geom type="cylinder" 
                          fromto="0 {current_height/2} 0  {-current_width/2} {-current_height/2} 0"
                          size="0.002"
                          rgba=".4 .4 .8 1"/>
                          
                    <!-- Fill in the triangular face -->
                    <geom type="box"
                          pos="0 0 {section_length/2}"
                          size="{current_width/3} {current_height/3} {section_length/2}"
                          rgba=".4 .4 .8 0.8"/>
                    
                    <!-- Cable routing points -->
                    <site name="{name}_flexor_site_{i}_a" 
                          pos="{current_width/2} {-current_height/4} {-section_length/4}" 
                          size="0.001" rgba="1 0 0 0.5"/>
                    <site name="{name}_flexor_site_{i}_b" 
                          pos="{current_width/2} {-current_height/4} {section_length/4}" 
                          size="0.001" rgba="1 0 0 0.5"/>
                    
                    <site name="{name}_extensor_site_{i}_a" 
                          pos="{-current_width/2} {current_height/4} {-section_length/4}" 
                          size="0.001" rgba="0 1 0 0.5"/>
                    <site name="{name}_extensor_site_{i}_b" 
                          pos="{-current_width/2} {current_height/4} {section_length/4}" 
                          size="0.001" rgba="0 1 0 0.5"/>
            """
        
        # Close all nested bodies
        for i in range(n_sections-1):
            current_body += "</body>\n"
        
        # Add the chain to the base and close base body
        xml += current_body + "</body></body>\n"
        
        return xml
    
    def generate_xml(self, n_sections):
        """Generate the complete XML model string"""
        xml = f"""
        <mujoco>
            <option gravity="0 0 -9.81">
                <flag contact="enable" constraint="enable"/>
            </option>
            
            <default>
                <joint armature="0.1" damping="0.1" limited="true"/>
                <motor ctrllimited="true" ctrlrange="-1 1"/>
                <geom condim="4" friction="1 0.1 0.1"/>
                <tendon width="0.001"/>
            </default>

            <worldbody>
                <!-- Fixed mount, rotated 180° around X-axis -->
                <body name="mount" pos="0 0 {self.mount_height}" euler="180 0 0">
                    <joint name="mount_joint" type="hinge" axis="0 0 1" range="-0.0001 0.0001" damping="1000" stiffness="1000"/>
                    <geom type="box" size="0.05 0.05 0.01" rgba="0.5 0.5 0.5 1"/>
                    
                    <!-- Gripper base attached to mount -->
                    <body name="gripper_base" pos="0 0 -0.02">
                        <geom type="box" size="0.03 0.03 0.02" rgba="0.7 0.7 0.7 1"/>
                        <!-- Rotated fingers by 90° -->
                        {self.generate_finger_xml("finger1", 0.02, 0, 90, n_sections)}
                        {self.generate_finger_xml("finger2", -0.02, 0, 270, n_sections)}
                    </body>
                </body>

                <!-- Objects to grasp at different positions -->
                <body name="sphere1" pos="0 0 0.15">
                    <joint type="free"/>
                    <geom type="sphere" size="0.02" rgba="1 0 0 1" mass="0.1"/>
                </body>
                
                <body name="sphere2" pos="0.2 0 0.15">
                    <joint type="free"/>
                    <geom type="sphere" size="0.02" rgba="0 1 0 1" mass="0.1"/>
                </body>
                
                <body name="sphere3" pos="-0.2 0 0.15">
                    <joint type="free"/>
                    <geom type="sphere" size="0.02" rgba="0 0 1 1" mass="0.1"/>
                </body>

                <!-- Ground plane -->
                <geom type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
            </worldbody>

            <tendon>
                {self.generate_actuator_xml(n_sections)}
            </tendon>

            <actuator>
                <!-- Add actuators for each finger -->
                <motor name="finger1_flexor_motor" tendon="finger1_flexor" gear="1000" 
                       ctrllimited="true" ctrlrange="0 200" forcerange="0 200"/>
                <motor name="finger1_extensor_motor" tendon="finger1_extensor" gear="1000" 
                       ctrllimited="true" ctrlrange="0 200" forcerange="0 200"/>
                <motor name="finger2_flexor_motor" tendon="finger2_flexor" gear="1000" 
                       ctrllimited="true" ctrlrange="0 200" forcerange="0 200"/>
                <motor name="finger2_extensor_motor" tendon="finger2_extensor" gear="1000" 
                       ctrllimited="true" ctrlrange="0 200" forcerange="0 200"/>
            </actuator>
        </mujoco>
        """
        return xml

    def add_object(self, shape, size, position, mass=0.1, rgba=(1,0,0,1)):
        """Add a new object to the simulation"""
        # Create new body and geom
        body = self.model.body_add()
        geom = self.model.geom_add()
        joint = self.model.joint_add()

        # Set body properties
        self.model.body_pos[body] = position
        self.model.body_parentid[body] = 0  # Attach to world

        # Set geom properties
        self.model.geom_type[geom] = {'sphere': 0, 'box': 1, 'cylinder': 2}[shape]
        self.model.geom_size[geom] = size
        self.model.geom_rgba[geom] = rgba
        self.model.geom_mass[geom] = mass
        self.model.geom_bodyid[geom] = body

        # Add free joint
        self.model.joint_type[joint] = mujoco.mjtJoint.mjJNT_FREE
        self.model.joint_bodyid[joint] = body

    def init_viewer(self):
        """Initialize GLFW viewer"""
        glfw.init()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
    def run_simulation(self):
        """Run the simulation with GLFW viewer"""
        if not glfw.init():
            return
        
        # Create window
        window = glfw.create_window(1200, 900, "SpiRob Gripper", None, None)
        if not window:
            glfw.terminate()
            return
        
        glfw.make_context_current(window)
        
        # Initialize camera
        cam = mujoco.MjvCamera()
        cam.distance = 2.0
        cam.azimuth = 90
        cam.elevation = -20
        
        # Add mouse controller variables
        button_left = False
        button_middle = False
        button_right = False
        lastx = 0
        lasty = 0

        def mouse_button_callback(window, button, act, mods):
            nonlocal button_left, button_middle, button_right
            
            # Update button states
            if button == glfw.MOUSE_BUTTON_LEFT:
                button_left = act == glfw.PRESS
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                button_middle = act == glfw.PRESS
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                button_right = act == glfw.PRESS
            
            # Update mouse position
            nonlocal lastx, lasty
            lastx, lasty = glfw.get_cursor_pos(window)

        def mouse_move_callback(window, xpos, ypos):
            nonlocal lastx, lasty
            
            # Compute mouse displacement
            dx = xpos - lastx
            dy = ypos - lasty
            
            # Update camera based on mouse movement
            if button_right:
                # Rotate view
                cam.azimuth += dx * 0.5
                cam.elevation = np.clip(cam.elevation - dy * 0.5, -90, 90)
            elif button_left:
                # Move camera target
                forward = np.array([np.cos(np.deg2rad(cam.azimuth)) * np.cos(np.deg2rad(cam.elevation)),
                                  np.sin(np.deg2rad(cam.azimuth)) * np.cos(np.deg2rad(cam.elevation)),
                                  np.sin(np.deg2rad(cam.elevation))])
                right = np.array([-np.sin(np.deg2rad(cam.azimuth)),
                                 np.cos(np.deg2rad(cam.azimuth)),
                                 0])
                up = np.cross(forward, right)
                cam.lookat[0] += 0.01 * cam.distance * (right[0]*dx - up[0]*dy)
                cam.lookat[1] += 0.01 * cam.distance * (right[1]*dx - up[1]*dy)
                cam.lookat[2] += 0.01 * cam.distance * (right[2]*dx - up[2]*dy)
            elif button_middle:
                # Adjust zoom
                cam.distance = cam.distance * (1.0 + dy*0.01)
                
            # Remember cursor position
            lastx = xpos
            lasty = ypos

        def scroll_callback(window, xoffset, yoffset):
            # Adjust zoom with scroll wheel
            cam.distance = cam.distance * (1.0 - yoffset*0.1)

        # Set callbacks
        glfw.set_mouse_button_callback(window, mouse_button_callback)
        glfw.set_cursor_pos_callback(window, mouse_move_callback)
        glfw.set_scroll_callback(window, scroll_callback)

        # Modified force parameters
        force_step = 5.0
        max_force = 100.0
        flexor_forces = [0.0, 0.0]
        extensor_forces = [0.0, 0.0]
        gripper_pos = [0, 0, self.mount_height]  # Current gripper position

        def keyboard_callback(window, key, scancode, action, mods):
            nonlocal flexor_forces, extensor_forces, gripper_pos
            
            if action != glfw.PRESS and action != glfw.REPEAT:
                return
                
            # Camera controls
            if key == glfw.KEY_LEFT:
                cam.azimuth += 2.0
            elif key == glfw.KEY_RIGHT:
                cam.azimuth -= 2.0
            elif key == glfw.KEY_UP:
                cam.elevation = min(cam.elevation + 2.0, 90)
            elif key == glfw.KEY_DOWN:
                cam.elevation = max(cam.elevation - 2.0, -90)
            elif key == glfw.KEY_Z:
                cam.distance *= (1 - 0.01)
            elif key == glfw.KEY_X:
                cam.distance *= (1 + 0.01)
            elif key == glfw.KEY_R:
                # Reset everything
                cam.azimuth = 90
                cam.elevation = -20
                cam.distance = 2.0
                cam.lookat[:] = [0, 0, 0.5]
                flexor_forces[:] = [0.0, 0.0]
                extensor_forces[:] = [0.0, 0.0]
            
            # Debug print to verify key presses
            print(f"Key pressed: {key}")
            
            # Object positioning controls
            if key == glfw.KEY_1:
                gripper_pos[:] = [0, 0, self.mount_height]  # Above object 1
                mount_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mount')
                self.model.body_pos[mount_id] = gripper_pos
            elif key == glfw.KEY_2:
                gripper_pos[:] = [0.2, 0, self.mount_height]  # Above object 2
                mount_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mount')
                self.model.body_pos[mount_id] = gripper_pos
            elif key == glfw.KEY_3:
                gripper_pos[:] = [-0.2, 0, self.mount_height]  # Above object 3
                mount_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mount')
                self.model.body_pos[mount_id] = gripper_pos
            
            # Gripper control
            elif key == glfw.KEY_O:  # Open gripper
                print("Opening gripper")
                for i in range(2):
                    extensor_forces[i] = max_force
                    flexor_forces[i] = 0
            elif key == glfw.KEY_C:  # Close gripper
                print("Closing gripper")
                for i in range(2):
                    flexor_forces[i] = max_force
                    extensor_forces[i] = 0
            elif key == glfw.KEY_SPACE:  # Reset gripper forces
                print("Resetting gripper")
                for i in range(2):
                    flexor_forces[i] = max_force/2
                    extensor_forces[i] = max_force/2

        # Add keyboard callback
        glfw.set_key_callback(window, keyboard_callback)

        # Initialize scene and context
        scene = mujoco.MjvScene(self.model, maxgeom=10000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # Add visualization options
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False  # Hide contact forces
        opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False       # Hide tendons
        opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False          # Hide center of mass
        opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False        # Hide joint axes

        def render_info(window, scene):
            """Render contact forces and joint angles"""
            # Get window dimensions for text placement
            width, height = glfw.get_window_size(window)
            
            # Prepare text overlay
            overlay = mujoco.MjrRect(0, 0, width, height)
            
            # Clear overlay (pass individual rgba values)
            mujoco.mjr_rectangle(overlay, 0, 0, 0, 0)
            
            # Update force display to show both cables
            force_text = "Cable Forces:\n"
            for i in range(2):
                force_text += f"Finger {i}: Flex: {flexor_forces[i]:.1f}, Extend: {extensor_forces[i]:.1f}\n"
            
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                overlay,
                force_text,
                "",
                context
            )
            
            # Display joint angles for each finger
            y_offset = 0
            for i in range(2):
                angles = []
                for j in range(self.model.nu // 3):
                    # Find joint by name using mujoco.mj_name2id
                    joint_name = f"finger_{i}_joint_{j}"
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id >= 0:  # Valid joint found
                        angle = np.degrees(self.data.qpos[joint_id])
                        angles.append(angle)
                
                if angles:  # Only display if we found angles
                    # Calculate average bend
                    avg_bend = np.mean(angles)
                    max_bend = np.max(np.abs(angles))
                    
                    angle_text = f"Finger {i}: Avg bend: {avg_bend:.1f}°, Max bend: {max_bend:.1f}°"
                    mujoco.mjr_overlay(
                        mujoco.mjtFont.mjFONT_NORMAL,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        mujoco.MjrRect(10, height - 30 - y_offset, width, height),
                        angle_text,
                        "",
                        context
                    )
                    y_offset += 20
            
            # Display contact forces if any
            contacts = []
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                # Use contact.dist instead of force_
                force_mag = abs(contact.dist)  # Use absolute value of distance as force magnitude
                if force_mag > 1e-3:  # Filter out tiny forces
                    pos = contact.pos
                    contacts.append(f"Contact {i}: pos [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], force: {force_mag:.2f}N")
            
            # Display contact information
            if contacts:
                contact_text = "Contacts:\n" + "\n".join(contacts)
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    overlay,
                    contact_text,
                    "",
                    context
                )

        # Simulation loop
        while not glfw.window_should_close(window):
            time_prev = time.time()
            
            # Apply forces to actuators with direct force control
            for i in range(2):  # Changed from 3 to 2
                flexor_idx = i * 2
                extensor_idx = i * 2 + 1
                
                # Apply forces directly
                self.data.ctrl[flexor_idx] = flexor_forces[i]
                self.data.ctrl[extensor_idx] = extensor_forces[i]
            
            # Reset actuator activations
            mujoco.mj_forward(self.model, self.data)
            
            # Step simulation multiple times for stability
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
            
            # Get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            
            # Update scene and render
            mujoco.mjv_updateScene(
                self.model, 
                self.data, 
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                scene
            )
            
            # Render scene
            mujoco.mjr_render(viewport, scene, context)
            
            # Render information overlay
            render_info(window, scene)
            
            # Swap buffers
            glfw.swap_buffers(window)
            glfw.poll_events()
            
            # Control loop timing
            time_until_next = time_prev + 0.02 - time.time()
            if time_until_next > 0:
                time.sleep(time_until_next)
        
        glfw.terminate()
    
    def apply_control(self, finger_idx, controls):
        """Apply control signals to a finger"""
        start_idx = finger_idx * self.model.nu // 3
        end_idx = (finger_idx + 1) * self.model.nu // 3
        self.data.ctrl[start_idx:end_idx] = controls

    def get_finger_state(self, finger_idx):
        """Get comprehensive state of a finger"""
        n_joints = self.model.nu // 3
        angles = []
        for j in range(n_joints):
            joint_name = f"finger_{finger_idx}_joint_{j}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                angle = np.degrees(self.data.qpos[joint_id])
                angles.append(angle)
        
        return {
            'angles': angles,
            'avg_bend': np.mean(angles) if angles else 0.0,
            'max_bend': np.max(np.abs(angles)) if angles else 0.0,
            'cable_force': self.data.actuator_force[finger_idx]
        }
    
    def get_contact_info(self):
        """Get information about current contacts"""
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Use contact.dist for force magnitude
            force_mag = abs(contact.dist)
            if force_mag > 1e-3:
                contacts.append({
                    'position': contact.pos,
                    'force': force_mag,  # Just use magnitude since we don't have direction
                    'magnitude': force_mag,
                    'geom1': self.model.geom_id2name(contact.geom1),
                    'geom2': self.model.geom_id2name(contact.geom2)
                })
        return contacts

    def generate_actuator_xml(self, n_sections):
        """Generate XML for the tendons"""
        xml = ""
        
        # Add tendons for each finger
        for i in range(2):  # 2 fingers
            # Flexor tendon
            xml += f"""
                <spatial name="finger{i+1}_flexor" width="0.001" rgba="1 0 0 1">
                    <site site="finger{i+1}_flexor_base"/>
            """
            for j in range(n_sections):
                xml += f"""
                    <site site="finger{i+1}_flexor_site_{j}_a"/>
                    <site site="finger{i+1}_flexor_site_{j}_b"/>
                """
            xml += """
                </spatial>
            """
            
            # Extensor tendon
            xml += f"""
                <spatial name="finger{i+1}_extensor" width="0.001" rgba="0 1 0 1">
                    <site site="finger{i+1}_extensor_base"/>
            """
            for j in range(n_sections):
                xml += f"""
                    <site site="finger{i+1}_extensor_site_{j}_a"/>
                    <site site="finger{i+1}_extensor_site_{j}_b"/>
                """
            xml += """
                </spatial>
            """
        
        return xml

if __name__ == "__main__":
    # Create and run the gripper simulation
    gripper = SpiRobGripper(n_sections=10)
    gripper.run_simulation() 