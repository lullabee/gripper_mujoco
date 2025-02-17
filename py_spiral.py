from matplotlib import pyplot as plt
import numpy as np

# Create array of angles from 0 to 6π (3 full rotations)
theta = np.linspace(0, 3*2*np.pi, 1000)

# Parameters for the logarithmic spiral
a = .3  # Initial radius (spiral starting point)
b = .22 # Growth rate of the spiral (controls how quickly it expands)
# Note: b = 1/tan(80°) would give a spiral with constant 80° angle between 
# radius and tangent

# Basic logarithmic spiral equation: r = a * e^(b*θ)
# - r is the radius at angle θ
# - a is the starting radius
# - b controls how tightly the spiral is wound
r = a * np.exp(b*theta)

# Calculate radius for a companion spiral
# This averages the radius of current point with point 2π ahead
# Creates a second spiral that's "between" the main spiral's loops
rc = (a * np.exp(b*theta) + a * np.exp(b*(theta + 2*np.pi))) / 2

def spiral(a, b, theta):
    """
    Calculate points for both spirals at a given angle.
    Returns points that form a segment between the spirals.
    """
    # Main spiral radius
    r = a * np.exp(b*theta)
    # Companion spiral radius (averaged with point 2π ahead)
    rc = (a * np.exp(b*theta) + a * np.exp(b*(theta + 2*np.pi))) / 2
    
    # Convert polar (r,θ) to Cartesian (x,y) for main spiral
    x1 = np.cos(theta) * r
    y1 = np.sin(theta) * r

    # Convert polar to Cartesian for companion spiral
    x2 = np.cos(theta) * rc
    y2 = np.sin(theta) * rc
    return x1, y1, x2, y2

# Create segment by calculating points at θ and θ+30°
x1, y1, x2, y2 = spiral(a, b, 0)  # Points at θ=0
x3, y3, x4, y4 = spiral(a, b, np.deg2rad(30))  # Points at θ=30°

# Print coordinates of quadrilateral segment
print(f'x:{np.array([x1,x2,x4,x3,x1])},\ny:{np.array([y1,y2,y4,y3,y1])}')

# Plot the quadrilateral segment
plt.plot([x1,x2,x4,x3,x1], [y1,y2,y4,y3,y1])
plt.axis('equal')

# Plot both spirals in polar coordinates
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r)   # Main spiral
ax.plot(theta, rc)  # Companion spiral

# Calculate growth factor over 30° segment
# This shows how much the spiral grows over the segment angle
print(f'Beta:{np.exp(b*np.deg2rad(30))}, {1/np.exp(b*np.deg2rad(30))}')

plt.show()