import glfw
import mujoco
import time
import numpy as np

class GripperSimulation:
    def __init__(self, gripper):
        self.gripper = gripper
        self.window = None
        self.scene = None
        self.context = None
        self.cam = None
        
        # Simulation state
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        self.force_step = 0.01
        self.max_force = 0.25
        self.flexor_forces = [0.0, 0.0]
        self.extensor_forces = [0.0, 0.0]
        self.gripper_pos = [0, 0, self.gripper.mount_height]
        self.levitated_object = None
        self.levitation_height = 0.2  # Height to levitate objects
        
    def init_visualization(self):
        """Initialize GLFW window and MuJoCo visualization"""
        if not glfw.init():
            return False
            
        self.window = glfw.create_window(1200, 900, "SpiRob Gripper", None, None)
        if not self.window:
            glfw.terminate()
            return False
            
        glfw.make_context_current(self.window)
        
        # Initialize camera
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 2.0
        self.cam.azimuth = 90
        self.cam.elevation = -20
        
        # Initialize scene and context
        self.scene = mujoco.MjvScene(self.gripper.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.gripper.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # Add visualization options
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        
        return True
        
    def setup_callbacks(self):
        """Setup input callbacks"""
        glfw.set_key_callback(self.window, self.keyboard_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        
    def keyboard_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if action != glfw.PRESS and action != glfw.REPEAT:
            return
            
        # Camera controls
        if key == glfw.KEY_LEFT:
            self.cam.azimuth += 2.0
        elif key == glfw.KEY_RIGHT:
            self.cam.azimuth -= 2.0
        elif key == glfw.KEY_UP:
            self.cam.elevation = min(self.cam.elevation + 2.0, 90)
        elif key == glfw.KEY_DOWN:
            self.cam.elevation = max(self.cam.elevation - 2.0, -90)
        elif key == glfw.KEY_Z:
            self.cam.distance *= (1 - 0.01)
        elif key == glfw.KEY_X:
            self.cam.distance *= (1 + 0.01)
        elif key == glfw.KEY_R:
            self.reset_view()
            
        # Object positioning controls
        elif key == glfw.KEY_1:
            self.move_to_object(0)
        elif key == glfw.KEY_2:
            self.move_to_object(1)
        elif key == glfw.KEY_3:
            self.move_to_object(2)
            
        # Gripper control
        elif key == glfw.KEY_O:
            self.gripper.open_gripper(self.max_force)
        elif key == glfw.KEY_C:
            self.gripper.close_gripper(self.max_force)
        elif key == glfw.KEY_SPACE:
            self.gripper.reset_gripper(self.max_force/2)
            
        # Teleport object control
        elif key == glfw.KEY_T:
            if self.levitated_object is None:
                # Find closest object below gripper
                closest_obj, qpos_adr = self.find_closest_object()
                if closest_obj is not None:
                    self.levitated_object = (closest_obj, qpos_adr)
                    self.teleport_object_to_gripper()
            else:
                # Release currently levitated object
                self.levitated_object = None
            
        # Individual tendon control
        elif key == glfw.KEY_E:  # Extend finger 1 (increase extensor)
            flexor_idx = 0
            extensor_idx = 1
            self.gripper.data.ctrl[extensor_idx] += self.force_step
            
        elif key == glfw.KEY_D:  # Contract finger 1 (increase flexor)
            flexor_idx = 0
            extensor_idx = 1
            self.gripper.data.ctrl[flexor_idx] += self.force_step
            
        elif key == glfw.KEY_I:  # Extend finger 2 (increase extensor)
            flexor_idx = 2
            extensor_idx = 3
            self.gripper.data.ctrl[extensor_idx] += self.force_step
            
        elif key == glfw.KEY_K:  # Contract finger 2 (increase flexor)
            flexor_idx = 2
            extensor_idx = 3
            self.gripper.data.ctrl[flexor_idx] += self.force_step
            
        # Relax tendons
        elif key == glfw.KEY_W:  # Relax finger 1
            flexor_idx = 0
            extensor_idx = 1
            self.gripper.data.ctrl[flexor_idx] = 0
            self.gripper.data.ctrl[extensor_idx] = 0
            
        elif key == glfw.KEY_S:  # Relax finger 2
            flexor_idx = 2
            extensor_idx = 3
            self.gripper.data.ctrl[flexor_idx] = 0
            self.gripper.data.ctrl[extensor_idx] = 0
            
    def mouse_button_callback(self, window, button, act, mods):
        """Handle mouse button input"""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.button_left = act == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.button_middle = act == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.button_right = act == glfw.PRESS
            
        self.lastx, self.lasty = glfw.get_cursor_pos(window)
        
    def mouse_move_callback(self, window, xpos, ypos):
        """Handle mouse movement"""
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        
        if self.button_right:
            self.cam.azimuth += dx * 0.5
            self.cam.elevation = np.clip(self.cam.elevation - dy * 0.5, -90, 90)
        elif self.button_left:
            self.move_camera(dx, dy)
        elif self.button_middle:
            self.cam.distance = self.cam.distance * (1.0 + dy*0.01)
            
        self.lastx = xpos
        self.lasty = ypos
        
    def scroll_callback(self, window, xoffset, yoffset):
        """Handle scroll input"""
        self.cam.distance = self.cam.distance * (1.0 - yoffset*0.1)
        
    def move_camera(self, dx, dy):
        """Move camera target based on mouse input"""
        forward = np.array([
            np.cos(np.deg2rad(self.cam.azimuth)) * np.cos(np.deg2rad(self.cam.elevation)),
            np.sin(np.deg2rad(self.cam.azimuth)) * np.cos(np.deg2rad(self.cam.elevation)),
            np.sin(np.deg2rad(self.cam.elevation))
        ])
        right = np.array([
            -np.sin(np.deg2rad(self.cam.azimuth)),
            np.cos(np.deg2rad(self.cam.azimuth)),
            0
        ])
        up = np.cross(forward, right)
        
        self.cam.lookat[0] += 0.01 * self.cam.distance * (right[0]*dx - up[0]*dy)
        self.cam.lookat[1] += 0.01 * self.cam.distance * (right[1]*dx - up[1]*dy)
        self.cam.lookat[2] += 0.01 * self.cam.distance * (right[2]*dx - up[2]*dy)
        
    def reset_view(self):
        """Reset camera to default position"""
        self.cam.azimuth = 90
        self.cam.elevation = -20
        self.cam.distance = 2.0
        self.cam.lookat[:] = [0, 0, 0.5]
        
    def move_to_object(self, obj_idx):
        """Move gripper above specified object"""
        positions = [[0, 0], [0.2, 0], [-0.2, 0]]
        self.gripper_pos[:2] = positions[obj_idx]
        self.gripper.set_position(self.gripper_pos)
        
    def render_info(self):
        """Render simulation information overlay"""
        width, height = glfw.get_window_size(self.window)
        overlay = mujoco.MjrRect(0, 0, width, height)
        
        # Clear overlay
        mujoco.mjr_rectangle(overlay, 0, 0, 0, 0)
        
        # Render states and forces
        self.render_finger_info(overlay, height)
        self.render_contact_info(overlay)
        
    def render_finger_info(self, overlay, height):
        """Render finger states and forces"""
        y_offset = 0
        for i in range(2):
            state = self.gripper.get_finger_state(i)
            info = f"Finger {i}: Avg bend: {state['avg_bend']:.1f}°, Max bend: {state['max_bend']:.1f}°"
            
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                mujoco.MjrRect(10, height - 30 - y_offset, overlay.width, overlay.height),
                info,
                "",
                self.context
            )
            y_offset += 20
            
    def render_contact_info(self, overlay):
        """Render contact information"""
        contacts = self.gripper.get_contact_info()
        if contacts:
            contact_text = "Contacts:\n" + "\n".join(
                f"Contact {i}: pos [{c['position'][0]:.2f}, {c['position'][1]:.2f}, {c['position'][2]:.2f}], "
                f"force: {c['force']:.2f}N"
                for i, c in enumerate(contacts)
            )
            
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                overlay,
                contact_text,
                "",
                self.context
            )
            
    def find_closest_object(self):
        """Find the closest object below the gripper"""
        gripper_pos = self.gripper_pos
        min_dist = float('inf')
        closest_obj = None
        
        # Get qpos indices for each object
        qpos_indices = {}
        for obj_name in ['sphere1', 'sphere2', 'sphere3']:
            body_id = mujoco.mj_name2id(self.gripper.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id >= 0:
                # Find the joint ID for this body
                for j in range(self.gripper.model.njnt):
                    if self.gripper.model.jnt_bodyid[j] == body_id:
                        qpos_adr = self.gripper.model.jnt_qposadr[j]
                        qpos_indices[obj_name] = (body_id, qpos_adr)
                        break
        
        # Find closest object
        for obj_name, (body_id, _) in qpos_indices.items():
            obj_pos = self.gripper.data.xpos[body_id]
            if obj_pos[2] < gripper_pos[2]:
                dist = np.sqrt((obj_pos[0] - gripper_pos[0])**2 + 
                              (obj_pos[1] - gripper_pos[1])**2)
                if dist < min_dist and dist < 0.1:  # Within 10cm radius
                    min_dist = dist
                    closest_obj = obj_name
        
        return closest_obj, qpos_indices.get(closest_obj) if closest_obj else None

    def teleport_object_to_gripper(self):
        """Move object to center of gripper"""
        if self.levitated_object is not None:
            obj_name, (body_id, qpos_adr) = self.levitated_object
            
            # Set position
            target_pos = [
                self.gripper_pos[0],
                self.gripper_pos[1],
                self.gripper_pos[2] - 0.1  # Slightly below gripper
            ]
            
            # Update position (first 3 elements of qpos for the free joint)
            self.gripper.data.qpos[qpos_adr:qpos_adr+3] = target_pos
            # Keep original orientation (next 4 elements are quaternion)
            self.gripper.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
            
            # Reset velocities
            body_vel_adr = body_id * 6  # 6 values per body (3 linear + 3 angular)
            self.gripper.data.qvel[body_vel_adr:body_vel_adr+6] = 0

    def run(self):
        """Run the simulation loop"""
        while not glfw.window_should_close(self.window):
            time_prev = time.time()
            
            # Step physics
            mujoco.mj_forward(self.gripper.model, self.gripper.data)
            for _ in range(5):
                mujoco.mj_step(self.gripper.model, self.gripper.data)
                
            # Render
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
            
            mujoco.mjv_updateScene(
                self.gripper.model,
                self.gripper.data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scene
            )
            
            mujoco.mjr_render(viewport, self.scene, self.context)
            self.render_info()
            
            # Update levitated object position if exists
            if self.levitated_object is not None:
                self.teleport_object_to_gripper()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            # Control timing
            time_until_next = time_prev + 0.02 - time.time()
            if time_until_next > 0:
                time.sleep(time_until_next)
                
        glfw.terminate() 
