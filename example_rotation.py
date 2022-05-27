# Import libraries
from numpy import array, cos, sin, vstack
from numpy.linalg import norm
from lib.convert import rad2deg, deg2rad
from lib.rotation import euler2dcm, dcm2euler, euler2quat, quat2euler, dcm2axis_ang, axis_ang2dcm, dcm2quat, \
    quat2dcm, qinv, qmult


# euler angles (example)
roll = deg2rad(10.0)
pitch = deg2rad(20.0)
yaw = deg2rad(30.0)

# euler to dcm
C = euler2dcm(roll, pitch, yaw)
print("euler2dcm:")
print(C, norm(C, 2))

# dcm to euler
roll, pitch, yaw = dcm2euler(C)
print("dcm2euler:")
print(rad2deg(roll), rad2deg(pitch), rad2deg(yaw))

# euler to quaternion
q = euler2quat(roll, pitch, yaw)
print("euler2quat:")
print(q, norm(q, 2))

# quaternion to euler
roll, pitch, yaw = quat2euler(q)
print("quat2euler:")
print(rad2deg(roll), rad2deg(pitch), rad2deg(yaw))

# dcm to axis_angle
theta, r = dcm2axis_ang(C)
print("DCM2axis_ang:")
print(theta, r.T)

# axis_angle to dcm
C = axis_ang2dcm(theta, r)
print("axis_ang2DCM:")
print(C, norm(C, 2))

# DCM to quaternion
q = dcm2quat(C)
print("dcm2quat:")
print(q, norm(q, 2))

# quaternion to DCM
C = quat2dcm(q)
print("quat2dcm:")
print(C, norm(C, 2))

# Rotation by quaternions (full example)

# Rotation angle
theta = deg2rad(30)

# Rotation axis (normalized)
r = array([[0],
           [0],
           [1]])

# Point
p1 = array([[1],
            [0],
            [0]])

# Point (quaternion representation)
p1q = vstack([0,
              p1])
print("Point p1:")
print(p1q, norm(p1q, 2))

# Define rotation quaternion
q = vstack([cos(theta/2),
           sin(theta/2)*r])
print("Rotation quaternion:")
print(q, norm(q, 2))

# Rotate p1 to p2
p2q = qmult(qmult(q, p1q), qinv(q))
print("Point p2:")
print(p2q, norm(p2q, 2))


# Corresponding dcm rotation

# Quaternion to dcm
C = quat2dcm(q)
print("quat2dcm:")
print(C, norm(C, 2))

# Rotate p1 to p2
p2 = C@p1
print("Point p2:")
print(p2, norm(p2, 2))
