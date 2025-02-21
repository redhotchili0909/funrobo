# Mini-Project I: Forward Kinematics

## Overview
This mini-project aims to:
- Strengthen your ability to apply the Denavit-Hartenberg (DH) convention for determining forward kinematics equations.
- Practice deriving and applying Jacobian matrices for robot manipulators (e.g., for resolved-rate motion control).
- Develop your proficiency in Python programming for robotics applications.

This assignment will count towards your **final course grade**. It is a team-based assignment with 2-3 students in a team.

## What to Do

**Problem 1**
- Derive the DH parameters and table for the 5DOF robot platform
- Derive the FPK equations and implement in software (with verification in the Viz tool)

**Problem 2**
- Derive the inverse jacobian matrix and implement a resolved-rate motion control (RRMC) with verification in the Viz tool

**Problem 3**
- Implement the resolved-rate motion control (RRMC) through gamepad control of the 5DOF robot hardware (**For instructions on how to use the Hiwonder robot platform, please** [**see here**](https://github.com/OlinCollege-FunRobo/hiwonder-armpi-pro))

<br />

<img src = "media/mini-project-1.png">

## What to Submit

A **PDF submission** with the following:
- Diagram of the robot arm with the coordinate frames clearly defined
- Derivation of the DH parameters and table and forward kinematics equations for the 5DOF robot platform
- Derivation of the jacobian matrix and inverse jacobian matrix for the robot arm
- Code showing all the functions you implemented
- Link to a screenshot video showing your verification using the viz tool for both forward position kinematics and velocity kinematics (see [here](https://github.com/OlinCollege-FunRobo/arm-kinematics-module/blob/main/resources/user_inputs.md) for user input instructions)
- Link to a video showing gamepad control of the physical 5DOF robot arm
- Link to your team's Github repository with a reasonably setup README
- **Individual reflections** on the following questions:
    - What did you learn from this? What did you not know before this assignment?
    - What was the most difficult aspect of the assignment?
    - What was the easiest or most straightforward aspect of the assignment?
    - How long did this assignment take? What took the most time (PC setup? Coding in Python? Exploring the questions?)?
    - What did you learn about arm forward kinematics that we didn't explicitly cover in class?
    - What more would you like to learn about arm forward kinematics?

