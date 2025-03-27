def tm_solver(theta_init):
    """Solves a transformation matrix and returns the matrix and its isolated position outputs.
    For given theta values:
    - Plugs in and solves transformation matrix.
    - Finds the end effector position from given theta values.

    Returns position values for the end effector.

    *******
    Keyword Arguments:
    *******
    theta_init: type=array; required. Initial guesses of theta
    """
    import math

    theta_0 = theta_init
    t1 = theta_0[0]
    t2 = theta_0[1]
    t3 = theta_0[2]
    t4 = theta_0[3]
    t5 = theta_0[4]
    a2 = 2
    a3 = 3
    d1 = 4
    d5 = 5

    n11 = math.cos(t1 + t2 + t3 + t4 + t5) + math.sin(t1 + t5)
    n12 = -math.sin(t5) * math.cos(t1 + t2 + t3 + t4) + math.sin(t1) * math.cos(t5)
    n13 = math.cos(t1) * math.sin(t2 + t3 + t4)
    n14 = math.cos(t1) * (
        d5 * math.sin(t2 + t3 + t4) + a3 * math.cos(t2 + t3) + a2 * math.cos(t2)
    )
    n21 = math.sin(t1) * math.cos(t2 + t3 + t4 + t5) - math.cos(t1) * math.sin(t5)
    n22 = -math.sin(t1 + t5) * math.cos(t2 + t3 + t4) - math.cos(t1 + t5)
    n23 = math.sin(t1) * math.sin(t2 + t3 + t4)
    n24 = math.sin(t1) * (
        d5 * math.sin(t2 + t3 + t4) + a3 * math.cos(t2 + t3) + a2 * math.cos(t2)
    )
    n31 = math.cos(t5) * math.sin(t2 + t3 + t4)
    n32 = -math.sin(t5) * math.sin(t2 + t3 + t4)
    n33 = -math.cos(t2 + t3 + t4)
    n34 = -d5 * math.cos(t2 + t3 + t4) + a3 * math.sin(t2 + t3) + a2 * math.sin(t2) + d1

    TM = [
        [n11, n21, n31, 0],
        [n12, n22, n32, 0],
        [n13, n23, n33, 0],
        [n14, n24, n34, 1],
    ]

    Pos = [round(n14, 3), round(n24, 3), round(n34, 3)]
    return [TM, Pos]


def inverse_jacobian(theta_init):

    # To find the angular and linear velocity of the end effector we must multiply the Jacobian matrix by the velocity of joint (i)
    # For visual clarity we isolated the linear and angular components of the Jacobian and then combined them

    # The linear and angular components of the Jacobian are each a 3x5 matrix containing information on the x,y and z axes locations and angles

    import numpy as np
    import math

    # Linear Velocity of Robot

    # Values from our DH table
    d1 = 0.155  # (m)
    d5 = 0.15  # (m)
    a2 = 0
    a3 = 0

    # Non-zero placeholder values of the servo angles in radians
    # Non-zero values prevent issues with singularities
    theta1 = theta_init[0]
    theta2 = theta_init[1]
    theta3 = theta_init[2]
    theta4 = theta_init[3]

    # Compiling the linear components of the Jacobian Matrix
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

    # Compiling the angular components of the Jacobian Matrix

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

    # We found the Linear velocity of frame(i) by cross multiplying the angle change at theta(i)
    # by the difference in the position of the end effector frame and frame (i)

    # Below are the components that make up the linear velocity portion of the Jacobian Matrix
    J11 = np.multiply(z0, (O5 - Oo))
    J12 = np.multiply(z1, (O5 - O1))
    J13 = np.multiply(z2, (O5 - O2))
    J14 = np.multiply(z3, (O5 - O3))
    J15 = np.multiply(z4, (O5 - O4))

    # Assembling the linear components of the Jacobian Matrix
    Jacobian_Linear = np.hstack([J11, J12, J13, J14, J15])

    # We found the values of the angular velocity at each joint by differentiating our FPK equations
    # The components below give the x,y and z angular velocities for each joint of the robot

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

    return Jacobian_Linear


def num_analysis(xd, theta_0, n):
    """Finds the angle values for each joint of a robot given the desired end-effector position.
    For a given transformation matrix and desired end-effector position:
    - Defines a function g which quantifies the error in the EE(end-effector)
    position at each iteration compared to the desired EE position.
    - Finds the value of the inverse(if invertable) jacobian matrix,
    or pseudoinverse(if not invertable) jacobian matrix.
    - Uses the inverse jacobian and error to find the next angle values in the correct direction
    - Sets current angle values to the new values
    - Iterates until the error value is zero
    - Counts each iteration, and if the number of iterations surpasses n prints error statement

    Returns angle values for each joint in the robot that will result in the desired EE position.

    *******
    Keyword Arguments:
    *******

    xd: type=array; required.
    An array of the x, y, and z values of the desired end effector position.

    theta_0: type=array; required. An array of initial angle value guessed.

    n: type=int; required.
    The max number of iterations that can be tried before the function gives up.
    """
    import numpy as np

    theta_i = theta_0
    ee_p = np.array(tm_solver(theta_i)[1])
    g = np.array(xd) - ee_p
    i = 0
    while np.all(g) != 0:
        psinvj = inverse_jacobian(theta_i)
        psinvj_reshaped = psinvj.reshape((5, 3))
        ## with a pseudoinverse/inverse depending on its dimension
        theta_i = theta_i + np.dot(psinvj_reshaped, g)  # Future angle value
        ee_p = np.array(tm_solver(theta_i)[1])
        g = np.array(xd) - ee_p
        i = i + 1
        print(i)
        print(g)
        if i >= n:
            break
    else:
        print("The joint angles satisfy the desired end-effector position.")


theta_inits = [12, 13, 15, 19, 25]
num_analysis([3, 4, 5], theta_inits, 10)
