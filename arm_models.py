from math import sqrt, sin, cos, atan, atan2, degrees, pi
import numpy as np
from matplotlib.figure import Figure
from helper_fcns.utils import EndEffector, rotm_to_euler, euler_to_rotm, check_joint_limits, dh_to_matrix, near_zero, wraptopi
import time

PI = 3.1415926535897932384
# np.set_printoptions(precision=3)

class Robot:
    """
    Represents a robot manipulator with various kinematic configurations.
    Provides methods to calculate forward kinematics, inverse kinematics, and velocity kinematics.
    Also includes methods to visualize the robot's motion and state in 3D.

    Attributes:
        num_joints (int): Number of joints in the robot.
        ee_coordinates (list): List of end-effector coordinates.
        robot (object): The robot object (e.g., TwoDOFRobot, ScaraRobot, etc.).
        origin (list): Origin of the coordinate system.
        axes_length (float): Length of the axes for visualization.
        point_x, point_y, point_z (list): Lists to store coordinates of points for visualization.
        show_animation (bool): Whether to show the animation or not.
        plot_limits (list): Limits for the plot view.
        fig (matplotlib.figure.Figure): Matplotlib figure for 3D visualization.
        sub1 (matplotlib.axes._subplots.Axes3DSubplot): Matplotlib 3D subplot.
    """

    def __init__(self, type='2-dof', show_animation: bool=True):
        """
        Initializes a robot with a specific configuration based on the type.

        Args:
            type (str, optional): Type of robot (e.g., '2-dof', 'scara', '5-dof'). Defaults to '2-dof'.
            show_animation (bool, optional): Whether to show animation of robot movement. Defaults to True.
        """
        if type == '2-dof':
            self.num_joints = 2
            self.ee_coordinates = ['X', 'Y']
            self.robot = TwoDOFRobot()
        
        elif type == 'scara':
            self.num_joints = 3
            self.ee_coordinates = ['X', 'Y', 'Z', 'Theta']
            self.robot = ScaraRobot()

        elif type == '5-dof':
            self.num_joints = 5
            self.ee_coordinates = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
            self.robot = FiveDOFRobot()
        
        self.origin = [0., 0., 0.]
        self.axes_length = 0.04
        self.point_x, self.point_y, self.point_z = [], [], []
        self.waypoint_x, self.waypoint_y, self.waypoint_z = [], [], []
        self.waypoint_rotx, self.waypoint_roty, self.waypoint_rotz = [], [], []
        self.show_animation = show_animation
        self.plot_limits = [0.65, 0.65, 0.8]

        if self.show_animation:
            self.fig = Figure(figsize=(12, 10), dpi=100)
            self.sub1 = self.fig.add_subplot(1,1,1, projection='3d') 
            self.fig.suptitle("Manipulator Kinematics Visualization", fontsize=16)

        # initialize figure plot
        self.init_plot()

    
    def init_plot(self):
        """Initializes the plot by calculating the robot's points and calling the plot function."""
        self.robot.calc_robot_points()
        self.plot_3D()

    
    def update_plot(self, pose=None, angles=None, soln=0, numerical=False):
        """
        Updates the robot's state based on new pose or joint angles and updates the visualization.

        Args:
            pose (EndEffector, optional): Desired end-effector pose for inverse kinematics.
            angles (list, optional): Joint angles for forward kinematics.
            soln (int, optional): The inverse kinematics solution to use (0 or 1).
            numerical (bool, optional): Whether to use numerical inverse kinematics.
        """
        if pose is not None: # Inverse kinematics case
            if not numerical:
                self.robot.calc_inverse_kinematics(pose, soln=soln)
            else:
                self.robot.calc_numerical_ik(pose, tol=0.005, ilimit=1500)
        elif angles is not None: # Forward kinematics case
            self.robot.calc_forward_kinematics(angles, radians=False)
        else:
            return
        self.plot_3D()


    def move_velocity(self, vel):
        """
        Moves the robot based on a given velocity input.

        Args:
            vel (list): Velocity input for the robot.
        """
        self.robot.calc_velocity_kinematics(vel)
        self.plot_3D()
        

    def draw_line_3D(self, p1, p2, format_type: str = "k-"):
        """
        Draws a 3D line between two points.

        Args:
            p1 (list): Coordinates of the first point.
            p2 (list): Coordinates of the second point.
            format_type (str, optional): The format of the line. Defaults to "k-".
        """
        self.sub1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], format_type)


    def draw_ref_line(self, point, axes=None, ref='xyz'):
        """
        Draws reference lines from a given point along specified axes.

        Args:
            point (list): The coordinates of the point to draw from.
            axes (matplotlib.axes, optional): The axes on which to draw the reference lines.
            ref (str, optional): Which reference axes to draw ('xyz', 'xy', or 'xz'). Defaults to 'xyz'.
        """
        line_width = 0.7
        if ref == 'xyz':
            axes.plot([point[0], self.plot_limits[0]],
                      [point[1], point[1]],
                      [point[2], point[2]], 'b--', linewidth=line_width)    # X line
            axes.plot([point[0], point[0]],
                      [point[1], self.plot_limits[1]],
                      [point[2], point[2]], 'b--', linewidth=line_width)    # Y line
            axes.plot([point[0], point[0]],
                      [point[1], point[1]],
                      [point[2], 0.0], 'b--', linewidth=line_width)         # Z line
        elif ref == 'xy':
            axes.plot([point[0], self.plot_limits[0]],
                      [point[1], point[1]], 'b--', linewidth=line_width)    # X line
            axes.plot([point[0], point[0]],
                      [point[1], self.plot_limits[1]], 'b--', linewidth=line_width)    # Y line
        elif ref == 'xz':
            axes.plot([point[0], self.plot_limits[0]],
                      [point[2], point[2]], 'b--', linewidth=line_width)    # X line
            axes.plot([point[0], point[0]],
                      [point[2], 0.0], 'b--', linewidth=line_width)         # Z line


    def plot_waypoints(self):
        """
        Plots the waypoints in the 3D visualization
        """
        # draw the points
        self.sub1.plot(self.waypoint_x, self.waypoint_y, self.waypoint_z, 'or', markersize=8)


    def update_waypoints(self, waypoints: list):
        """
        Updates the waypoints into a member variable
        """
        for i in range(len(waypoints)):
            self.waypoint_x.append(waypoints[i][0])
            self.waypoint_y.append(waypoints[i][1])
            self.waypoint_z.append(waypoints[i][2])
            # self.waypoint_rotx.append(waypoints[i][3])
            # self.waypoint_roty.append(waypoints[i][4])
            # self.waypoint_rotz.append(waypoints[i][5])


    def plot_3D(self):
        """
        Plots the 3D visualization of the robot, including the robot's links, end-effector, and reference frames.
        """        
        self.sub1.cla()
        self.point_x.clear()
        self.point_y.clear()
        self.point_z.clear()

        EE = self.robot.ee

        # draw lines to connect the points
        for i in range(len(self.robot.points)-1):
            self.draw_line_3D(self.robot.points[i], self.robot.points[i+1])

        # draw the points
        for i in range(len(self.robot.points)):
            self.point_x.append(self.robot.points[i][0])
            self.point_y.append(self.robot.points[i][1])
            self.point_z.append(self.robot.points[i][2])
        self.sub1.plot(self.point_x, self.point_y, self.point_z, marker='o', markerfacecolor='m', markersize=12)


        # draw the waypoints
        self.plot_waypoints()

        # draw the EE
        self.sub1.plot(EE.x, EE.y, EE.z, 'bo')
        # draw the base reference frame
        self.draw_line_3D(self.origin, [self.origin[0] + self.axes_length, self.origin[1], self.origin[2]], format_type='r-')
        self.draw_line_3D(self.origin, [self.origin[0], self.origin[1] + self.axes_length, self.origin[2]], format_type='g-')
        self.draw_line_3D(self.origin, [self.origin[0], self.origin[1], self.origin[2] + self.axes_length], format_type='b-')
        # draw the EE reference frame
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[0], format_type='r-')
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[1], format_type='g-')
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[2], format_type='b-')
        # draw reference / trace lines
        self.draw_ref_line([EE.x, EE.y, EE.z], self.sub1, ref='xyz')

        # add text at bottom of window
        pose_text = "End-effector Pose:      [ "
        pose_text += f"X: {round(EE.x,4)},  "
        pose_text += f"Y: {round(EE.y,4)},  "
        pose_text += f"Z: {round(EE.z,4)},  "
        pose_text += f"RotX: {round(EE.rotx,4)},  "
        pose_text += f"RotY: {round(EE.roty,4)},  "
        pose_text += f"RotZ: {round(EE.rotz,4)}  "
        pose_text += " ]"

        theta_text = "Joint Positions (deg/m):     ["
        for i in range(self.num_joints):
            theta_text += f" {round(np.rad2deg(self.robot.theta[i]),2)}, "
        theta_text += " ]"

        print(theta_text)
        print(pose_text)
        
        textstr = pose_text + "\n" + theta_text
        self.sub1.text2D(0.2, 0.02, textstr, fontsize=13, transform=self.fig.transFigure)

        self.sub1.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        self.sub1.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        self.sub1.set_zlim(0, self.plot_limits[2])
        self.sub1.set_xlabel('x [m]')
        self.sub1.set_ylabel('y [m]')




class TwoDOFRobot():
    """
    Represents a 2-degree-of-freedom (DOF) robot arm with two joints and one end effector.
    Includes methods for calculating forward kinematics (FPK), inverse kinematics (IPK),
    and velocity kinematics (VK).

    Attributes:
        l1 (float): Length of the first arm segment.
        l2 (float): Length of the second arm segment.
        theta (list): Joint angles.
        theta_limits (list): Joint limits for each joint.
        ee (EndEffector): The end effector object.
        points (list): List of points representing the robot's configuration.
        num_dof (int): The number of degrees of freedom (2 for this robot).
    """

    def __init__(self):
        """
        Initializes a 2-DOF robot with default arm segment lengths and joint angles.
        """
        self.l1 = 0.30  # Length of the first arm segment
        self.l2 = 0.25  # Length of the second arm segment

        self.theta = [0.0, 0.0]  # Joint angles (in radians)
        self.theta_limits = [[-PI, PI], [-PI + 0.261, PI - 0.261]]  # Joint limits

        self.ee = EndEffector()  # The end-effector object
        self.num_dof = 2  # Number of degrees of freedom
        self.points = [None] * (self.num_dof + 1)  # List to store robot points


    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculates the forward kinematics for the robot based on the joint angles.

        Args:
            theta (list): Joint angles.
            radians (bool, optional): Whether the angles are in radians or degrees. Defaults to False.
        """
        if not radians:
            # Convert degrees to radians if the input is in degrees
            self.theta[0] = np.deg2rad(theta[0])
            self.theta[1] = np.deg2rad(theta[1])
        else:
            self.theta = theta

        # Ensure that the joint angles respect the joint limits
        for i, th in enumerate(self.theta):
            self.theta[i] = np.clip(th, self.theta_limits[i][0], self.theta_limits[i][1])

        # Update the robot configuration (i.e., the positions of the joints and end effector)
        self.calc_robot_points()


    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculates the inverse kinematics (IK) for a given end effector position.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            soln (int, optional): The solution branch to use. Defaults to 0 (first solution).
        """
        
        x, y = EE.x, EE.y
        l1, l2 = self.l1, self.l2

        # Move robot slightly out of zero configuration to avoid singularity
        if all(th == 0.0 for th in self.theta):
            self.theta = [self.theta[i] + np.random.rand()*0.01 for i in range(self.num_dof)]
        
        try:
            if soln == 0:
                # Solution 0: Calculate joint angles using cosine rule
                beta = np.arccos((l1**2 + l2**2 - x**2 - y**2) / (2*l1*l2))
                self.theta[1] = PI - beta
                c2 = np.cos(self.theta[1])
                s2 = np.sin(self.theta[1])
                
                alpha = atan2((l2*s2), (l1 + l2*c2))
                gamma = atan2(y, x)
                self.theta[0] = gamma - alpha
            elif soln == 1:
                # Solution 1: Alternative calculation for theta_1 and theta_2
                beta = np.arccos((l1**2 + l2**2 - x**2 - y**2) / (2*l1*l2))
                self.theta[1] = beta - PI
                c2 = np.cos(self.theta[1])
                s2 = np.sin(self.theta[1])

                alpha = atan2((l2*s2), (l1 + l2*c2))
                gamma = atan2(y, x)
                self.theta[0] = gamma - alpha
            else:
                raise ValueError("Invalid IK solution specified!")
            
            if not check_joint_limits(self.theta, self.theta_limits):
                print(f"\n [ERROR] Joint limits exceeded! \n \
                      Desired joints are {self.theta} \n \
                      Joint limits are {self.theta_limits}")
                raise ValueError
            
        except RuntimeWarning:
            print("[ERROR] Joint limits exceeded (runtime issue).")
        
        # Calculate robot points based on the updated joint angles
        self.calc_robot_points()


    def calc_velocity_kinematics(self, vel: list):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy].
        """
        
        # move robot slightly out of zeros singularity
        if all(th == 0.0 for th in self.theta):
            self.theta = [self.theta[i] + np.random.rand()*0.02 for i in range(self.num_dof)]
        
        # Calculate joint velocities using the inverse Jacobian
        vel = vel[:2]  # Consider only the first two components of the velocity
        thetadot = self.inverse_jacobian() @ vel

        # print(f'thetadot: {thetadot}')

        # (Corrective measure) Ensure joint velocities stay within limits
        self.thetadot_limits = [[-PI, PI], [-PI, PI]]
        thetadot = np.clip(thetadot, [limit[0] for limit in self.thetadot_limits], [limit[1] for limit in self.thetadot_limits])
        
        # print(f'Limited thetadot: {thetadot}')

        # Update the joint angles based on the velocity
        self.theta[0] += 0.02 * thetadot[0]
        self.theta[1] += 0.02 * thetadot[1]

        # Ensure joint angles stay within limits
        self.theta = np.clip(self.theta, [limit[0] for limit in self.theta_limits], [limit[1] for limit in self.theta_limits])

        # Update robot points based on the new joint angles
        self.calc_robot_points()


    def jacobian(self, theta: list = None):
        """
        Returns the Jacobian matrix for the robot. If theta is not provided, 
        the function will use the object's internal theta attribute.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.

        Returns:
            np.ndarray: The Jacobian matrix (2x2).
        """
        # Use default values if arguments are not provided
        if theta is None:
            theta = self.theta
        
        return np.array([
            [-self.l1 * sin(theta[0]) - self.l2 * sin(theta[0] + theta[1]), -self.l2 * sin(theta[0] + theta[1])],
            [self.l1 * cos(theta[0]) + self.l2 * cos(theta[0] + theta[1]), self.l2 * cos(theta[0] + theta[1])]
        ])


    def inverse_jacobian(self):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        J = self.jacobian()
        print(f'Determinant of J: {np.linalg.det(J)}')
        # return np.linalg.inv(self.jacobian())
        return np.linalg.pinv(self.jacobian())
     
    
    def solve_forward_kinematics(self, theta: list, radians=False):
        """
        Evaluates the forward kinematics based on the joint angles.

        Args:
            theta (list): The joint angles [theta_1, theta_2].
            radians (bool, optional): Whether the input angles are in radians (True) or degrees (False).
                                      Defaults to False (degrees).

        Returns:
            list: The end effector position [x, y].
        """
        if not radians:
            theta[0] = np.deg2rad(theta[0])
            theta[1] = np.deg2rad(theta[1])

        # Compute forward kinematics
        x = self.l1 * cos(theta[0]) + self.l2 * cos(theta[0] + theta[1])
        y = self.l1 * sin(theta[0]) + self.l2 * sin(theta[0] + theta[1])

        return [x, y]
    

    def calc_robot_points(self):
        """
        Calculates the positions of the robot's joints and the end effector.

        Updates the `points` list, storing the coordinates of the base, shoulder, elbow, and end effector.
        """
        # Base position
        self.points[0] = [0.0, 0.0, 0.0]
        # Shoulder joint
        self.points[1] = [self.l1 * cos(self.theta[0]), self.l1 * sin(self.theta[0]), 0.0]
        # Elbow joint
        self.points[2] = [self.l1 * cos(self.theta[0]) + self.l2 * cos(self.theta[0] + self.theta[1]),
                          self.l1 * sin(self.theta[0]) + self.l2 * sin(self.theta[0] + self.theta[1]), 0.0]

        # Update end effector position
        self.ee.x = self.points[2][0]
        self.ee.y = self.points[2][1]
        self.ee.z = self.points[2][2]
        self.ee.rotz = self.theta[0] + self.theta[1]

        # End effector axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = np.array([cos(self.theta[0] + self.theta[1]), sin(self.theta[0] + self.theta[1]), 0]) * 0.075 + self.points[2]
        self.EE_axes[1] = np.array([-sin(self.theta[0] + self.theta[1]), cos(self.theta[0] + self.theta[1]), 0]) * 0.075 + self.points[2]
        self.EE_axes[2] = np.array([0, 0, 1]) * 0.075 + self.points[2]


class ScaraRobot():
    """
    A class representing a SCARA (Selective Compliance Assembly Robot Arm) robot.
    This class handles the kinematics (forward, inverse, and velocity kinematics) 
    and robot configuration, including joint limits and end-effector calculations.
    """

    
    def __init__(self):
        """
        Initializes the SCARA robot with its geometry, joint variables, and limits.
        Sets up the transformation matrices and robot points.
        """
        # Geometry of the robot (link lengths in meters)
        self.l1 = 0.35  # Base to 1st joint
        self.l2 = 0.18  # 1st joint to 2nd joint
        self.l3 = 0.15  # 2nd joint to 3rd joint
        self.l4 = 0.30  # 3rd joint to 4th joint (tool or end-effector)
        self.l5 = 0.12  # Tool offset

        # Joint variables (angles in radians)
        self.theta = [0.0, 0.0, 0.0]

        # Joint angle limits (min, max) for each joint
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi + 0.261, np.pi - 0.261],
            [0, self.l1 + self.l3 - self.l5]
        ]

        # End-effector (EE) object to store EE position and orientation
        self.ee = EndEffector()

        # Number of degrees of freedom and number of points to store robot configuration
        self.num_dof = 3
        self.num_points = 7
        self.points = [None] * self.num_points

        # Transformation matrices (DH parameters and resulting transformation)
        self.DH = np.zeros((5, 4))  # Denavit-Hartenberg parameters (theta, d, a, alpha)
        self.T = np.zeros((self.num_dof, 4, 4))  # Transformation matrices

    
    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            theta (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).
        """
        if not radians:
            self.theta[0] = np.deg2rad(theta[0])
            self.theta[1] = np.deg2rad(theta[1])
            self.theta[2] = theta[2]  # No need to convert z-axis theta
        else:
            self.theta = theta

        # Apply joint limits after updating joint angles
        for i, th in enumerate(self.theta):
            self.theta[i] = np.clip(th, self.theta_limits[i][0], self.theta_limits[i][1])

        # DH parameters for each joint
        ee_home = self.l3 - self.l5
        self.DH[0] = [self.theta[0], self.l1, self.l2, 0]
        self.DH[1] = [self.theta[1], ee_home, self.l4, 0]
        self.DH[2] = [0, -self.theta[2], 0, np.pi]

        # Calculate transformation matrices for each joint
        for i in range(self.num_dof):
            self.T[i] = dh_to_matrix(self.DH[i])

        # Calculate robot points (e.g., end-effector position)
        self.calc_robot_points()


    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculate Inverse Kinematics (IK) based on the input end-effector coordinates.

        Args:
            EE (EndEffector): End-effector object containing desired position (x, y, z).
            soln (int): Solution index (0 or 1), for multiple possible IK solutions.
        """
        x, y, z = EE.x, EE.y, EE.z
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        # Slightly perturb the robot from singularity at zero configuration
        if all(th == 0.0 for th in self.theta):
            self.theta = [self.theta[i] + np.random.rand() * 0.01 for i in range(self.num_dof)]

        try:
            # Select inverse kinematics solution
            if soln == 0:
                # Calculate theta_3
                self.theta[2] = l1 + l3 - l5 - z

                # Using the cosine rule, calculate theta_2
                beta = np.arccos((l2**2 + l4**2 - x**2 - y**2) / (2 * l2 * l4))
                self.theta[1] = np.pi - beta

                # Using trigonometry, find theta_1
                c2 = np.cos(self.theta[1])
                s2 = np.sin(self.theta[1])
                alpha = np.arctan2(l4 * s2, l2 + l4 * c2)
                gamma = np.arctan2(y, x)
                self.theta[0] = gamma - alpha

            elif soln == 1:
                # Alternate solution for theta_1 and theta_2
                self.theta[2] = l1 + l3 - l5 - z
                beta = np.arccos((l2**2 + l4**2 - x**2 - y**2) / (2 * l2 * l4))
                self.theta[1] = beta - np.pi

                c2 = np.cos(self.theta[1])
                s2 = np.sin(self.theta[1])
                alpha = np.arctan2(l4 * s2, l2 + l4 * c2)
                gamma = np.arctan2(y, x)
                self.theta[0] = gamma - alpha

            else:
                raise ValueError("Invalid IK solution specified!")

            # Check joint limits and ensure validity
            if not check_joint_limits(self.theta, self.theta_limits):
                print(f"\n [ERROR] Joint limits exceeded! \n \
                      Desired joints are {self.theta} \n \
                      Joint limits are {self.theta_limits}")
                raise ValueError

        except Exception as e:
            print(f"Error in Inverse Kinematics: {e}")
            return

        # Recalculate Forward Kinematics to update the robot's configuration
        self.calc_forward_kinematics(self.theta, radians=True)


    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate velocity kinematics and update joint velocities.

        Args:
            vel (array): Linear velocities (3D) of the end-effector.
        """
        # Ensure small deviation from singularity at the zero configuration
        if all(th == 0.0 for th in self.theta):
            self.theta = [self.theta[i] + np.random.rand() * 0.01 for i in range(self.num_dof)]

        # Compute the inverse of the Jacobian to update joint velocities
        thetadot = self.inverse_jacobian() @ vel

        # Update joint angles based on computed joint velocities
        self.theta[0] += 0.02 * thetadot[0]
        self.theta[1] += 0.02 * thetadot[1]
        self.theta[2] -= 0.01 * thetadot[2]

        # Apply joint limits after updating joint angles
        for i, th in enumerate(self.theta):
            self.theta[i] = np.clip(th, self.theta_limits[i][0], self.theta_limits[i][1])

        # Recalculate robot points based on updated joint angles
        self.calc_robot_points()
  

    def jacobian(self):
        """
        Compute the Jacobian matrix for the robot's end-effector.

        Returns:
            numpy.ndarray: 3x3 Jacobian matrix.
        """
        ee_home = self.l3 - self.l5
        self.DH[0] = [self.theta[0], self.l1, self.l2, 0]
        self.DH[1] = [self.theta[1], ee_home, self.l4, 0]
        self.DH[2] = [0, -self.theta[2], 0, np.pi]

        # Update transformation matrices
        for i in range(self.num_dof):
            self.T[i] = dh_to_matrix(self.DH[i])

        # Extract transformation matrices
        T0_1 = self.T[0]
        T0_2 = self.T[0] @ self.T[1]
        T0_3 = self.T[0] @ self.T[1] @ self.T[2]
        O0 = np.array([0, 0, 0, 1])

        jacobian = np.zeros((3, 3))

        # calculate Jv(1)
        # z0 X r, r = (Oc - O0)
        r = (T0_3 @ O0 - O0)[:3]
        # z0 = T0_1[:3,:3] @ np.array([0, 0, 1])
        z0 = np.array([0, 0, 1])
        Jv1 = np.linalg.cross(z0, r)

        # calculate Jv(2)
        r = (T0_3 @ O0 - T0_1 @ O0)[:3]
        z1 = T0_1[:3,:3] @ np.array([0, 0, 1])
        Jv2 = np.linalg.cross(z1, r)

        # calculate Jv(3)
        r = (T0_3 @ O0 - T0_2 @ O0)[:3]
        z2 = T0_2[:3,:3] @ np.array([0, 0, 1])
        Jv3 = z2

        jacobian[:,0] = Jv1
        jacobian[:,1] = Jv2
        jacobian[:,2] = Jv3

        return jacobian


    def inverse_jacobian(self):
        """
        Compute the inverse of the Jacobian matrix.

        Returns:
            numpy.ndarray: Inverse Jacobian matrix.
        """
        return np.linalg.inv(self.jacobian())


    def calc_robot_points(self):
        """
        Calculate the main robot points (links and end-effector position) using the current joint angles.
        Updates the robot's points array and end-effector position.
        """
        # Calculate transformation matrices for each joint and end-effector
        self.points[0] = np.array([0, 0, 0, 1])
        self.points[1] = np.array([0, 0, self.l1, 1])
        self.points[2] = self.T[0]@self.points[0]
        self.points[3] = self.points[2] + np.array([0, 0, self.l3, 1])
        self.points[4] = self.T[0]@self.T[1]@self.points[0] + np.array([0, 0, self.l5, 1])
        self.points[5] = self.T[0]@self.T[1]@self.points[0]
        self.points[6] = self.T[0]@self.T[1]@self.T[2]@self.points[0]

        self.EE_axes = self.T[0]@self.T[1]@self.T[2]@np.array([0.075, 0.075, 0.075, 1])
        self.T_ee = self.T[0]@self.T[1]@self.T[2]

        # End-effector (EE) position and axes
        self.ee.x = self.points[-1][0]
        self.ee.y = self.points[-1][1]
        self.ee.z = self.points[-1][2]
        rpy = rotm_to_euler(self.T_ee[:3,:3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy
        
        # EE coordinate axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = self.T_ee[:3,0] * 0.075 + self.points[-1][0:3]
        self.EE_axes[1] = self.T_ee[:3,1] * 0.075 + self.points[-1][0:3]
        self.EE_axes[2] = self.T_ee[:3,2] * 0.075 + self.points[-1][0:3]


class FiveDOFRobot:
    """
    A class to represent a 5-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (5 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """
    
    def __init__(self):
        """Initialize the robot parameters and joint limits."""
        # Link lengths
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.155, 0.099, 0.095, 0.055, (0.105 - 0.16) # from hardware measurements
        
        # Joint angles (initialized to zero)
        self.theta = [0, 0, 0, 0, 0]
        
        # Joint limits (in radians)
        self.theta_limits = [
            [-np.pi, np.pi], 
            [-np.pi/3, np.pi], 
            [-np.pi+np.pi/12, np.pi-np.pi/4], 
            [-np.pi+np.pi/12, np.pi-np.pi/12], 
            [-np.pi, np.pi]
        ]

        self.thetadot_limits = [
            [-np.pi*2, np.pi*2], 
            [-np.pi*2, np.pi*2], 
            [-np.pi*2, np.pi*2], 
            [-np.pi*2, np.pi*2], 
            [-np.pi*2, np.pi*2]
        ]
        
        # End-effector object
        self.ee = EndEffector()
        
        # Robot's points
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)
        
        # Denavit-Hartenberg parameters and transformation matrices
        self.DH = np.zeros((5, 4))
        self.T = np.zeros((self.num_dof, 4, 4))

    
    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate forward kinematics based on the provided joint angles.
        
        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        if not radians:
            # Convert degrees to radians
            self.theta = np.deg2rad(theta)
        else:
            self.theta = theta
        
        # Apply joint limits
        self.theta = [np.clip(th, self.theta_limits[i][0], self.theta_limits[i][1]) 
                      for i, th in enumerate(self.theta)]

        # Set the Denavit-Hartenberg parameters for each joint
        self.DH[0] = [self.theta[0], self.l1, 0, np.pi/2]
        self.DH[1] = [self.theta[1] + np.pi/2, 0, self.l2, np.pi]
        self.DH[2] = [self.theta[2], 0, self.l3, np.pi]
        self.DH[3] = [self.theta[3] - np.pi/2, 0, 0, -np.pi/2]
        self.DH[4] = [self.theta[4], self.l4 + self.l5, 0, 0]

        # Compute the transformation matrices
        for i in range(self.num_dof):
            self.T[i] = dh_to_matrix(self.DH[i])
        
        # Calculate robot points (positions of joints)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE, soln=0):
        """
        Calculate inverse kinematics analytically to determine the joint angles based on end-effector position.
        
        1. Calculate the wrist center position from the desired end-effector position
        2. Find candidate values for joint angles θ1, θ2, and θ3 (geometric approach)
        3. For each combination of θ1, θ2, θ3, solve for remaining joints θ4 and θ5
        4. Filter solutions based on joint limits
        5. Verify solutions by forward kinematics and select the best matching one
        
        Args:
            EE: EndEffector object containing desired position and orientation (EE.x, EE.y, EE.z, EE.rotx, EE.roty, EE.rotz).
            soln: Optional parameter to choose between multiple valid solutions (0 for first solution, other values for alternatives).
        """
        start_time = time.time()  # Start timing for analytical IK

        # STEP 1: Convert the desired end-effector orientation from Euler angles to rotation matrix
        # This represents the orientation of the end-effector frame relative to the base frame
        R_05 = euler_to_rotm((EE.rotx, EE.roty, EE.rotz))

        # STEP 2: Calculate the wrist center position using the end-effector position and orientation
        # The wrist center is located at a distance (l4+l5) from the end-effector along its z-axis
        # p_wrist = p_EE - d * z_EE (where d = l4 + l5 and z_EE is the end-effector z-axis direction)
        EE_pos = np.array([EE.x, EE.y, EE.z]).reshape(3, 1)  # End-effector position as column vector
        z_EE = np.array([0, 0, 1]).reshape(3, 1)             # Z-axis of end-effector frame in base frame coordinates
        p_wrist = EE_pos - (self.l4 + self.l5) * (R_05 @ z_EE)  # Matrix multiplication to get wrist center

        # Extract the coordinates of the wrist center
        x = p_wrist[0, 0]
        y = p_wrist[1, 0]
        z = p_wrist[2, 0]
        print("Wrist center:", x, y, z)

        # STEP 3: Calculate possible values for θ3 using the law of cosines
        # The distance L between the base of the arm (after accounting for link 1's height) and wrist center
        # forms a triangle with links l2 and l3, allowing us to use the law of cosines
        L = sqrt((x**2 + y**2) + ((z - self.l1) ** 2))  # Distance from base to wrist center
        
        # Using law of cosines: l2^2 + l3^2 - 2*l2*l3*cos(θ3) = L^2
        # Rearranging: cos(θ3) = (L^2 - l2^2 - l3^2) / (-2*l2*l3)
        cosang3 = (L**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        sinang3 = np.sqrt(1 - cosang3**2)  # sin(θ3) = ±√(1-cos^2(θ3))
        
        # There are two possible solutions for θ3 due to the ± in the sin calculation
        angle_3_candidates = [
            np.arctan2(sinang3, cosang3),    # "Elbow up" configuration
            np.arctan2(-sinang3, cosang3)    # "Elbow down" configuration
        ]

        # STEP 4: Calculate possible values for θ1
        # θ1 represents the rotation around the base z-axis
        # The wrist center projects onto the xy-plane, giving us atan2(y, x)
        # We need to consider two solutions:
        angle_1_candidates = [
            wraptopi(np.arctan2(y, x)),            # "Front" configuration
            wraptopi(np.pi + np.arctan2(y, x))     # "Back" configuration (180° rotated)
        ]

        # STEP 5: Calculate possible values for θ2
        # θ2 is calculated based on the projection of the wrist center in a plane
        # We need to consider the angle α between links l2 and L and the angle ψ between L and the xy-plane
        
        # α is the angle in the triangle formed by l2, l3, and L
        # Using law of cosines: cos(α) = (L^2 + l2^2 - l3^2) / (2*L*l2)
        alpha = np.acos((L**2 + self.l2**2 - self.l3**2) / (2 * L * self.l2))
        
        # ψ is the angle between L and the xy-plane
        # cos(ψ) = (z - l1) / L
        psi = np.acos((z - self.l1) / L)
        
        # There are multiple possible combinations for θ2 depending on the configurations
        th2_vals = [-alpha - psi, alpha + psi, alpha - psi, -alpha + psi]
        
        # Create a list to store all potential solutions before applying joint limits
        raw_solutions = []
        sol_index = 0

        # STEP 6: Generate all possible solutions by trying combinations of θ1, θ2, and θ3
        # For each combination, calculate θ4 and θ5 to achieve the desired end-effector orientation
        for th1 in angle_1_candidates:
            for th3 in angle_3_candidates:
                for th2 in th2_vals:
                    # Build the DH parameter table for the first 3 joints to calculate the rotation matrix R_03
                    DH_3 = np.zeros((3, 4))
                    # DH parameters: [θ, d, a, α]
                    DH_3[0] = [th1,           self.l1, 0,        np.pi/2]  # Joint 1
                    DH_3[1] = [th2+np.pi/2,   0,       self.l2,  np.pi  ]  # Joint 2
                    DH_3[2] = [th3,           0,       self.l3,  np.pi  ]  # Joint 3

                    # Calculate the homogeneous transformation matrix from base to joint 3
                    H_03 = np.eye(4)  # Initialize with identity matrix
                    for i in range(3):
                        H_03 = H_03 @ dh_to_matrix(DH_3[i])  # Multiply transformation matrices

                    # Extract the rotation component from H_03
                    R_03 = H_03[:3, :3]
                    
                    # Calculate the rotation from joint 3 to end-effector (R_35) using R_03 and R_05
                    # R_03 * R_35 = R_05, therefore R_35 = R_03^T * R_05
                    R_35 = R_03.T @ R_05

                    # STEP 7: Calculate θ4 and θ5 from R_35 components
                    # θ4 and θ5 are calculated using the θ & α elements of the R_35 matrix
                    th4 = np.arctan2(R_35[1, 2], R_35[0, 2])
                    th5 = np.arctan2(-R_35[2, 0], -R_35[2, 1])

                    sol_index += 1

                    # Add the complete joint angle solution to our list
                    raw_solutions.append((th1, th2, th3, th4, th5))

        print("\nTotal raw solutions (before limit checks):", len(raw_solutions))

        # STEP 8: Filter solutions based on joint limits
        # Check each solution against the robot's joint angle limits
        valid_solutions = []
        for (th1, th2, th3, th4, th5) in raw_solutions:
            angles = [th1, th2, th3, th4, th5]
            in_range = True
            
            for i, ang in enumerate(angles):
                lo, hi = self.theta_limits[i]  # Get the lower and upper limits for joint i
                if ang < lo or ang > hi:
                    in_range = False  # If any angle exceeds its limits, mark solution as invalid
                    break
                    
            # If all angles are within limits, add this solution to valid solutions
            if in_range:
                valid_solutions.append((th1, th2, th3, th4, th5))

        print("Number of solutions within joint limits:", len(valid_solutions))

        # STEP 9: Verify solutions using forward kinematics
        # For each valid solution, compute the resulting end-effector pose using forward kinematics
        # and calculate the error from the desired pose
        sol_count = 0
        solutions_with_error = []
        
        for solution in valid_solutions:
            sol_count += 1
            th1, th2, th3, th4, th5 = solution

            # Calculate the end-effector position and orientation for this solution
            EE_pos, EE_ori = self.solve_forward_kinematics(list(solution), radians=True)

            # Extract the components of the resulting end-effector pose
            x_sol, y_sol, z_sol = EE_pos[0], EE_pos[1], EE_pos[2]
            rx, ry, rz = EE_ori[0], EE_ori[1], EE_ori[2]

            # Calculate the Euclidean distance error between the achieved and desired positions
            pos_error = np.sqrt((x_sol - EE.x)**2 +
                        (y_sol - EE.y)**2 +
                        (z_sol - EE.z)**2)
            
            # Store the solution with its corresponding error
            solutions_with_error.append((solution, pos_error))
        
        # Sort solutions by increasing error (most accurate first)
        solutions_with_error.sort(key=lambda x: x[1])

        # Define a tolerance threshold for considering a solution as valid
        tolerance = 1e-6
        matched_solutions = []

        # STEP 10: Print solutions that meet the tolerance threshold
        for count, (sol, error) in enumerate(solutions_with_error):
            if abs(error) < tolerance:
                matched_solutions.append((sol, error))
                th1, th2, th3, th4, th5 = sol
                
                print(f"\nValid Solution #{count}:")
                print(f"  θ1 = {np.rad2deg(th1):7.2f} deg, "
                    f"θ2 = {np.rad2deg(th2):7.2f} deg, "
                    f"θ3 = {np.rad2deg(th3):7.2f} deg, "
                    f"θ4 = {np.rad2deg(th4):7.2f} deg, "
                    f"θ5 = {np.rad2deg(th5):7.2f} deg")
                    
                print(f"  --> End-Effector Pose (XYZ,RotXYZ): "
                    f"[ {x_sol:.5f}, {y_sol:.5f}, {z_sol:.5f}, "
                    f"{rx:.4f}, {ry:.4f}, {rz:.4f} ]")

        # STEP 11: Select the best solution
        # If no solutions meet the tolerance, warn and select the solution with the smallest error
        if not matched_solutions:
            print("No solutions within the tolerance; selecting the candidate with the smallest error instead.")
            best_sol, _ = solutions_with_error[0]  # Choose the solution with the smallest error
        else:
            if soln == 0:
                best_sol, _ = matched_solutions[0]
            else:
                best_sol, _ = matched_solutions[1] if len(matched_solutions) > 1 else matched_solutions[0]

        # STEP 12: Update the robot's state with the best solution
        self.calc_forward_kinematics(best_sol, radians=True)

        end_time = time.time()  # End timing for analytical IK
        print("Analytical IK computation time: {:.6f} seconds".format(end_time - start_time))


        
    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=50):
        """
        Calculate numerical inverse kinematics (IK) to move the robot's end-effector
        to the desired position (EE.x, EE.y, EE.z).
        
        Args:
            EE (EndEffector): desired end-effector pose (x, y, z).
            tol (float): position error tolerance in meters.
            ilimit (int): max number of iterations before giving up.
        """

        start_time = time.time()  # Start timing for numerical IK

        print("Start Numerical IK!")
        
        # Desired end-effector position
        xd = np.array([EE.x, EE.y, EE.z])
        
        # STEP 1: Initialize with current joint angles 
        theta_i = np.array(self.theta, dtype=float)
        
        # Begin the iterative solution process, limited by the maximum number of iterations
        # Each iteration will attempt to reduce the error between current and desired positions
        for i in range(ilimit):
            # STEP 2: Update robot state and compute forward kinematics
            # We apply our current joint angle guess to the robot model and calculate
            # where the end-effector would be with this configuration
            self.theta = theta_i.tolist()
            self.calc_forward_kinematics(self.theta, radians=True)
            
            # STEP 3: Compute position error vector (desired - current)
            # This measures how far our end-effector is from the target position
            # The goal of the algorithm is to minimize this error vector
            current_xyz = np.array([self.ee.x, self.ee.y, self.ee.z])
            g = xd - current_xyz
            
            # STEP 4: Check if we've reached the desired tolerance
            # We measure the magnitude (Euclidean norm) of the error vector
            # If it's small enough, we've found an acceptable solution
            if np.linalg.norm(g) < tol:
                print(f"Converged in {i} iterations.")
                print("Final joint angles (deg) =", np.round(np.degrees(theta_i), 3))
                print("Final end-effector position =", [round(v, 4) for v in current_xyz])
                end_time = time.time()  # End timing for numerical IK
                print("Numerical IK computation time: {:.6f} seconds".format(end_time - start_time))
                return
            
            # STEP 5: Compute the Jacobian matrix at current configuration
            # The Jacobian relates differential changes in joint angles to
            # differential changes in end-effector position
            J = self.jacobian(theta_i)
            
            # Calculate the pseudoinverse of the Jacobian matrix
            # This allows us to solve for joint angle changes given desired position changes
            J_pinv = np.linalg.pinv(J)
            
            # STEP 6: Update joint angles
            # θ_new = θ_current + J⁺ * error
            theta_i = theta_i + J_pinv @ g
            
            # STEP 7: Enforce joint limits by clipping values to allowed ranges
            for idx in range(self.num_dof):
                low, high = self.theta_limits[idx]
                theta_i[idx] = np.clip(theta_i[idx], low, high)
        
        # STEP 8: If we exit the loop without returning, we didn't find a solution
        # within the iteration limit
        print("No numerical solution found within iteration limit.")
        end_time = time.time()  # End timing for numerical IK for no solutions found
        print("Numerical IK computation time: {:.6f} seconds".format(end_time - start_time))


    def jacobian(self, theta: list = None):
        """
        Compute the Jacobian matrix for the current robot configuration.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.
        
        Returns:
            Jacobian matrix (3x5).
        """
        # Use default values if arguments are not provided
        if theta is None:
            theta = self.theta

        # Define DH parameters
        DH = np.zeros((5, 4))
        DH[0] = [theta[0], self.l1, 0, np.pi/2]
        DH[1] = [theta[1] + np.pi/2, 0, self.l2, np.pi]
        DH[2] = [theta[2], 0, self.l3, np.pi]
        DH[3] = [theta[3] - np.pi/2, 0, 0, -np.pi/2]
        DH[4] = [theta[4], self.l4 + self.l5, 0, 0]

        # Compute transformation matrices
        T = np.zeros((self.num_dof,4,4))
        for i in range(self.num_dof):
            T[i] = dh_to_matrix(DH[i])

        # Precompute transformation matrices for efficiency
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ T[i])

        # Define O0 for calculations
        O0 = np.array([0, 0, 0, 1])
        
        # Initialize the Jacobian matrix
        jacobian = np.zeros((3, self.num_dof))

        # Calculate the Jacobian columns
        for i in range(self.num_dof):
            T_curr = T_cumulative[i]
            T_final = T_cumulative[-1]
            
            # Calculate position vector r
            r = (T_final @ O0 - T_curr @ O0)[:3]

            # Compute the rotation axis z
            z = T_curr[:3, :3] @ np.array([0, 0, 1])

            # Compute linear velocity part of the Jacobian
            jacobian[:, i] = np.cross(z, r)

        # Replace near-zero values with zero, primarily for debugging purposes
        return near_zero(jacobian)
    
    
    def inverse_jacobian(self, pseudo=False):
        """
        Compute the inverse of the Jacobian matrix using either pseudo-inverse or regular inverse.
        
        Args:
            pseudo: Boolean flag to use pseudo-inverse (default is False).
        
        Returns:
            The inverse (or pseudo-inverse) of the Jacobian matrix.
        """

        J = self.jacobian()
        JT = np.transpose(J)
        manipulability_idx = np.sqrt(np.linalg.det(J @ JT))

        if pseudo:
            return np.linalg.pinv(self.jacobian())
        else:
            return np.linalg.inv(self.jacobian())
        
        
    def damped_inverse_jacobian(self, q = None, damping_factor=0.025):
        if q is not None:
            J = self.jacobian(q)
        else:
            J = self.jacobian()

        JT = np.transpose(J)
        I = np.eye(3)
        return JT @ np.linalg.inv(J @ JT + (damping_factor**2)*I)


    def dh_to_matrix(self, dh_params: list) -> np.ndarray:
        """Converts Denavit-Hartenberg parameters to a transformation matrix.

        Args:
            dh_params (list): Denavit-Hartenberg parameters [theta, d, a, alpha].

        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        theta, d, a, alpha = dh_params
        return np.array([
            [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
            [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])


    def solve_forward_kinematics(self, theta: list, radians=False):

        # Convert degrees to radians
        if not radians:
            for i in range(len(theta)):
                theta[i] = np.deg2rad(theta[i])

        # DH parameters = [theta, d, a, alpha]
        DH = np.zeros((5, 4))
        DH[0] = [theta[0],   self.l1,    0,       np.pi/2]
        DH[1] = [theta[1]+np.pi/2,   0,          self.l2, np.pi]
        DH[2] = [theta[2],   0,          self.l3, np.pi]
        DH[3] = [theta[3]-np.pi/2,   0,          0,       -np.pi/2]
        DH[4] = [theta[4],   self.l4+self.l5, 0, 0]

        T = np.eye(4)
        for i in range(self.num_dof):
            T = T @ dh_to_matrix(DH[i])
        
        pos = T @ np.array([0, 0, 0, 1])
        rpy = rotm_to_euler(T[:3, :3])
        return pos, rpy



    def calc_robot_points(self):
        """ Calculates the main arm points using the current joint angles """

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, 6):
            self.points[i] = T_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = T_cumulative[-1] @ np.array([0.075, 0.075, 0.075, 1])  # End-effector axes
        self.T_ee = T_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE = [self.ee.x, self.ee.y, self.ee.z]
        self.EE_axes = np.array([self.T_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)])