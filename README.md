# SpiRob Gripper Simulation

A MuJoCo simulation of a soft robotic gripper with tendon-driven actuation, based on the paper at https://arxiv.org/pdf/2303.09861v1.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

## Running the Simulation

Launch the simulation:

```bash
python spirob_mujoco.py
```

## Controls

### Camera Controls
- **Arrow Keys**: 
  - Left/Right: Rotate camera azimuth
  - Up/Down: Adjust camera elevation
- **Z/X**: Zoom in/out
- **R**: Reset camera to default view
- **Mouse Controls**:
  - Right Button + Drag: Rotate view
  - Left Button + Drag: Pan camera
  - Middle Button/Scroll: Zoom in/out

### Gripper Controls
- **Position**:
  - **1**: Move gripper above center object
  - **2**: Move gripper above right object
  - **3**: Move gripper above left object
- **Grasping**:
  - **O**: Open gripper (extend fingers)
  - **C**: Close gripper (flex fingers)
  - **Space**: Reset gripper to neutral position

## Display Information

The simulation window shows:
- Real-time cable forces for each finger
- Joint angles and average bend angles
- Contact forces during object interaction

## Features
- Two-finger soft robotic gripper
- Tendon-driven actuation (flexor/extensor pairs)
- Three test objects for grasping
- Downward-facing gripper mount
- Physics-based interaction with objects

## Troubleshooting

If you encounter OpenGL-related issues:
1. Ensure your graphics drivers are up to date
2. Try reinstalling PyOpenGL and PyOpenGL_accelerate
3. Check that your system supports OpenGL 3.3 or higher

## License

This simulation is provided for research and educational purposes.
