def tm_solver(theta_init):
    """Solves a transformation matrix and returns said matrix and its isolated position outputs.
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
    import numpy as np
    from numpy.linalg import multi_dot

    l1 = 0.155
    l2 = 0.099
    l3 = 0.095
    l4 = 0.055
    l5 = 0.105

    # Declare variables storing the angle input values and values from the DH table
    # DH parameters = [theta, d, a, alpha]
    DH = np.zeros((5, 4))
    DH[0] = [theta_init[0], l1, 0, math.pi / 2]
    DH[1] = [theta_init[1] + math.pi / 2, 0, l2, math.pi]
    DH[2] = [theta_init[2], 0, l3, math.pi]
    DH[3] = [theta_init[3] - math.pi / 2, 0, 0, -math.pi / 2]
    DH[4] = [theta_init[4], l4 + l5, 0, 0]

    # Declare variables storing the angle input values and values from the DH table
    theta = theta_init
    d = [DH[0][1], DH[1][1], DH[2][1], DH[3][1], DH[4][1]]
    a = [DH[0][2], DH[1][2], DH[2][2], DH[3][2], DH[4][2]]
    alpha = [DH[0][3], DH[1][3], DH[2][3], DH[3][3], DH[4][3]]
    num_matrices = 5  # Number of matrices in the array
    rows, cols = 4, 4  # Dimensions of each matrix
    matrix_array = np.array([np.zeros((rows, cols)) for _ in range(num_matrices)])
    TM = matrix_array

    # Finding each component of the transformation matrix and substituting variables
    for y in range(len(theta)):

        n11 = np.cos(theta[y])
        n12 = -np.sin(theta[y]) * np.cos(alpha[y])
        n13 = np.sin(theta[y]) * np.sin(alpha[y])
        n14 = a[y] * np.cos(theta[y])

        n21 = np.sin(theta[y])
        n22 = np.cos(theta[y]) * np.cos(alpha[y])
        n23 = -np.cos(theta[y]) * np.sin(alpha[y])
        n24 = a[y] * np.sin(theta[y])

        n31 = 0
        n32 = np.sin(alpha[y])
        n33 = np.cos(alpha[y])
        n34 = d[y]

        # Compiling the transformation matrix
        TM[y] = [
            [n11, n21, n31, 0],
            [n12, n22, n32, 0],
            [n13, n23, n33, 0],
            [n14, n24, n34, 1],
        ]

    Final_TM = multi_dot(TM)
    # Isolating the position outputs of the transformation matrix
    Pos = [round(Final_TM[3][0], 3), round(Final_TM[3][1], 3), round(Final_TM[3][2], 3)]

    # Return the matrix and the isolated position values
    return [Final_TM, Pos]


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


def num_analysis(xd, theta_init, n):
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

    theta_init: type=array; required. An array of initial angle value guessed.

    n: type=int; required.
    The max number of iterations that can be tried before the function gives up.
    """
    import numpy as np

    # Convert radians to degrees for visual output in terminal
    rad2deg = float(180 / np.pi)

    # Initialize theta_i
    theta_i = theta_init

    # Extract end-effector position
    ee_p = np.array(tm_solver(theta_i)[1])

    # Calculate error between desired and actual end-effector position
    g = np.array(xd) - ee_p

    # initialize iteration counter
    i = 0

    # Print the end-effector position from our initial guess of theta to guide the user to make better guesses
    print(ee_p)

    # While the end-effector is not at the desired position run this code
    while np.all(g) != 0:

        # Declare pseudoinverse of the jacobian matrix extracted from the function
        psinvj = inverse_jacobian(theta_i)

        # Reshape the matrix in order to be able to multiply it by the error array
        psinvj_reshaped = psinvj.reshape((5, 3))

        # Updating theta values at each iteration
        theta_i = theta_i + np.dot(psinvj_reshaped, g)  # Future angle value

        # Calculate end-effector position for the updated theta values
        ee_p = np.array(tm_solver(theta_i)[1])

        # Check to see if the stopping critera has been met
        g = np.array(xd) - ee_p

        # Update i to track the number of iterations
        i = i + 1

        # If the max number of iterations is surpassed
        if i >= n:
            # Print that there is no numerical solution for the given
            print("There is no numerical solution")

            # End the function to prevent an endless loop
            break

    else:

        theta_limits = [
            [-np.pi, np.pi],
            [-np.pi / 3, np.pi],
            [-np.pi + np.pi / 12, np.pi - np.pi / 4],
            [-np.pi + np.pi / 12, np.pi - np.pi / 12],
            [-np.pi, np.pi],
        ]

        for y, theta_limit in enumerate(theta_limits):
            if theta_i[y] < theta_limits[y][0] or theta_i[y] > theta_limits[y][1]:
                print("This angle is out of the range of the joint limits")
                break

        if theta_i[y] < theta_limits[y][0] or theta_i[y] > theta_limits[y][1]:
            pass
        else:

            theta_deg = np.round(theta_i * rad2deg, 3)
            print(
                "The joint angles (in degrees) "
                + str(theta_deg[0])
                + ", "
                + str(theta_deg[1])
                + ", "
                + str(theta_deg[2])
                + ", "
                + str(theta_deg[3])
                + ", and "
                + str(theta_deg[4])
                + " satisfy the desired end-effector position."
            )
            print("It took " + str(i) + " iterations to find the angle values")
            return np.round(theta_i, 3)


# Declare an array of initial angle guesses
theta_inits = [0.4, 0.5, 0.6, 0.2, 0.1]

# Declare an array including the desired end-effector position
desired_eep = [0.9, 0.445, 0.264]

# Attempt to find the theta values that will result in the desired end-effector position
theta_vals = num_analysis(desired_eep, theta_inits, 500)
print(theta_vals)
