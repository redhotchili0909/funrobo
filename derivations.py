import numpy as np
import math

# Linear and Angular Velocity of the Robot

# Values from our DH table
d1 = 0.155  # (m)
d5 = 0.15  # (m)
a2 = 0
a3 = 0

# Non-zero placeholder values of the servo angles in radians
# Non-zero values prevents issues with singularities while testing code
theta1 = 1.50098
theta2 = 2.26893
theta3 = 2.18166
theta4 = 1.0472


# Finding the location of joint(i)
Oo = np.array([[0], [0], [0]])
O1 = np.array([[0], [0], [d1]])
O2 = np.array(
    [
        [a2 * math.cos(theta1) * math.cos(theta2)],
        [a2 * math.sin(theta1) * math.cos(theta2)],
        [a2 * math.sin(theta2) + d1],
    ]
)
O3 = np.array(
    [
        [
            a3 * math.cos(theta1) * math.cos(theta2 + theta3)
            + a2 * math.cos(theta1) * math.cos(theta2)
        ],
        [
            a3 * math.sin(theta1) * math.cos(theta2 + theta3)
            + a2 * math.sin(theta1) * math.cos(theta2)
        ],
        [a3 * math.sin(theta2 + theta3) + a2 * math.sin(theta2) + d1],
    ]
)
O4 = np.array(
    [
        [
            a3 * math.cos(theta1) * math.cos(theta2 + theta3)
            + a2 * math.cos(theta1) * math.cos(theta2)
        ],
        [
            a3 * math.sin(theta1) * math.cos(theta2 + theta3)
            + a2 * math.sin(theta1) * math.cos(theta2)
        ],
        [a3 * math.sin(theta2 + theta3) + a2 * math.sin(theta2) + d1],
    ]
)

O5 = np.array(
    [
        [
            d5 * math.cos(theta1) * math.sin(theta2 + theta3 + theta4)
            + a3 * math.cos(theta1) * math.cos(theta2 + theta3)
            + a2 * math.cos(theta1) * math.cos(theta2)
        ],
        [
            d5 * math.sin(theta1) * math.sin(theta2 + theta3 + theta4)
            + a3 * math.sin(theta1) * math.cos(theta2 + theta3)
            + a2 * math.sin(theta1) * math.cos(theta2)
        ],
        [
            -d5 * math.cos(theta2 + theta3 + theta4)
            + a3 * math.sin(theta2 + theta3)
            + a2 * math.sin(theta2)
            + d1
        ],
    ]
)


# Finding the angle of joint(i)

z0 = np.array([[0], [0], [1]])
z1 = np.array([[math.sin(theta1)], [-math.cos(theta1)], [0]])
z3 = z2 = z1
z4 = np.array(
    [
        [math.cos(theta1) * math.sin(theta2 + theta3 + theta4)],
        [math.sin(theta1) * math.sin(theta2 + theta3 + theta4)],
        [-math.cos(theta2 + theta3 + theta4)],
    ]
)

# Finding the linear velocity of each frame by cross multiplying theta_prime(i) by the difference in the position of the end effector frame and frame (i)

# Below are the components that make up the linear components of the Jacobian Matrix
J11 = np.multiply(z0, (O5 - Oo))
J12 = np.multiply(z1, (O5 - O1))
J13 = np.multiply(z2, (O5 - O2))
J14 = np.multiply(z3, (O5 - O3))
J15 = np.multiply(z4, (O5 - O4))

# Assembling the linear components of the Jacobian Matrix
Jacobian_Linear = np.hstack([J11, J12, J13, J14, J15])


# We found the values of the angular velocity at each joint by differentiating our FPK equations
# The components below give the x,y and z angular velocities for each joint of the robot respectively

# Row one of the angular velocity components
Jm1 = [
    0,
    math.sin(theta1),
    math.sin(theta1),
    math.sin(theta1),
    math.cos(theta1) * math.sin(theta2 + theta3 + theta4),
]

# Row 2 of the angular velocity components
Jm2 = [
    0,
    -math.cos(theta1),
    -math.cos(theta1),
    -math.cos(theta1),
    math.sin(theta1) * math.sin(theta2 + theta3 + theta4),
]

# Row 3 of the angular velocity components
Jm3 = [1, 0, 0, 0, -math.cos(theta2 + theta3 + theta4)]

# Initializing an empty 3 x 5 matrix
Jacobian_Angular = np.zeros((3, 5))

# Populating the matrix of zeros with the angular velocity equations
Jacobian_Angular[0] = Jm1
Jacobian_Angular[1] = Jm2
Jacobian_Angular[2] = Jm3

# Combining the linear and angular components of the Jacobian Matrix
# This matrix allows us to find the linear and angular velocity of the end-effector based on theta_prime at each time step
Jacobian_Matrix = np.vstack([Jacobian_Linear, Jacobian_Angular])


# Finding the pseudo-inverse of the Jacobian Matrix
# Whatever velocity we get from the gamepad is multiplied by the Inverse Jacobian to obtain the change in angle
Jacob_Inverse = np.linalg.pinv(Jacobian_Matrix)