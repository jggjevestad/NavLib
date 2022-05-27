# Import libraries
from numpy import array, eye, vstack, sin, cos, sqrt, arcsin, arctan2, pi, trace
from numpy.linalg import norm


# Directional Cosine Matrix (DCM)
# x-axis rotation
def Rx(rx):
    return array([[1, 0, 0],
                  [0, cos(rx), -sin(rx)],
                  [0, sin(rx), cos(rx)]])


# y-axis rotation
def Ry(ry):
    return array([[cos(ry), 0, sin(ry)],
                  [0, 1, 0],
                  [-sin(ry), 0, cos(ry)]])


# z-axis rotation
def Rz(rz):
    return array([[cos(rz), -sin(rz), 0],
                  [sin(rz), cos(rz), 0],
                  [0, 0, 1]])


# dcm2euler
def dcm2euler(C):
    roll = arctan2(C[2, 1], C[2, 2])
    pitch = arctan2(-C[2, 0], sqrt(C[2, 1]**2 + C[2, 2]**2))
    yaw = arctan2(C[1, 0], C[0, 0])
    return roll, pitch, yaw


# euler2dcm
# Note: singular for pitch = +/- pi/2 (Gimbal lock)
# Sequence: yaw-pitch-roll (3-2-1)
def euler2dcm(roll, pitch, yaw):
    return Rz(yaw)@Ry(pitch)@Rx(roll)


# euler2quat
def euler2quat(roll, pitch, yaw):
    return array([[cos(roll/2)*cos(pitch/2)*cos(yaw/2) + sin(roll/2)*sin(pitch/2)*sin(yaw/2)],
                  [sin(roll/2)*cos(pitch/2)*cos(yaw/2) - cos(roll/2)*sin(pitch/2)*sin(yaw/2)],
                  [cos(roll/2)*sin(pitch/2)*cos(yaw/2) + sin(roll/2)*cos(pitch/2)*sin(yaw/2)],
                  [cos(roll/2)*cos(pitch/2)*sin(yaw/2) - sin(roll/2)*sin(pitch/2)*cos(yaw/2)]])


# quat2euler
def quat2euler(q):
    roll = arctan2(2*(q[2, 0]*q[3, 0] + q[0, 0]*q[1, 0]), 1 - 2*(q[1, 0]**2 + q[2, 0]**2))
    pitch = -arcsin(2*(q[1, 0]*q[3, 0] - q[0, 0]*q[2, 0]))
    yaw = arctan2(2*(q[1, 0]*q[2, 0] + q[0, 0]*q[3, 0]), 1 - 2*(q[2, 0]**2 + q[3, 0]**2))
    return roll, pitch, yaw


# Skew matrix
def skew(x):
    return array([[0, -x[2, 0], x[1, 0]],
                  [x[2, 0], 0, -x[0, 0]],
                  [-x[1, 0], x[0, 0], 0]])


# Axis-angle to dcm (Rodriguez formula)
# Note: Singularity for theta = 0 and theta = pi
def axis_ang2dcm(theta, r):
    Sr = skew(r)
    return eye(3) + sin(theta)*Sr + (1 - cos(theta))*Sr@Sr


# Axis-angle to DCM
def dcm2axis_ang(C):
    alpha = -array([[C[1, 2] - C[2, 1]],
                    [C[2, 0] - C[0, 2]],
                    [C[0, 1] - C[1, 0]]])
    theta = arctan2(norm(alpha), trace(C) - 1)
    r = alpha/norm(alpha)
    return theta, r


# Quaternion multiplication
def qmult(p, q):
    return array([[p[0, 0]*q[0, 0] - p[1, 0]*q[1, 0] - p[2, 0]*q[2, 0] - p[3, 0]*q[3, 0]],
                  [p[1, 0]*q[0, 0] + p[0, 0]*q[1, 0] - p[3, 0]*q[2, 0] + p[2, 0]*q[3, 0]],
                  [p[2, 0]*q[0, 0] + p[3, 0]*q[1, 0] + p[0, 0]*q[2, 0] - p[1, 0]*q[3, 0]],
                  [p[3, 0]*q[0, 0] - p[2, 0]*q[1, 0] + p[1, 0]*q[2, 0] + p[0, 0]*q[3, 0]]])


# Quaternion conjugate
def qconj(q):
    return vstack([q[0], -q[1:]])


# Quaternion inverse
def qinv(q):
    return qconj(q)/norm(q)**2


# DCM to quaternion
def dcm2quat(C):
    q0 = 1/2*sqrt(1 + trace(C))
    q = array([[q0],
               [(C[2, 1] - C[1, 2])/(4*q0)],
               [(C[0, 2] - C[2, 0])/(4*q0)],
               [(C[1, 0] - C[0, 1])/(4*q0)]])
    return q


# quaternion to DCM
def quat2dcm(q):
    return array([[1 - 2*(q[2, 0]**2 + q[3, 0]**2), 2*(q[1, 0]*q[2, 0] - q[0, 0]*q[3, 0]), 2*(q[1, 0]*q[3, 0] + q[0, 0]*q[2, 0])],
                  [2*(q[2, 0]*q[1, 0] + q[0, 0]*q[3, 0]), 1 - 2*(q[1, 0]**2 + q[3, 0]**2), 2*(q[2, 0]*q[3, 0] - q[0, 0]*q[1, 0])],
                  [2*(q[3, 0]*q[1, 0] - q[0, 0]*q[2, 0]), 2*(q[3, 0]*q[2, 0] + q[0, 0]*q[1, 0]), 1 - 2*(q[1, 0]**2 + q[2, 0]**2)]])


# Rotate from e-frame to g-frame (ned)
def Ce_g(lat, lon):
    return array([[-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)],
                  [-sin(lon), cos(lon), 0],
                  [-cos(lat)*cos(lon), -cos(lat)*sin(lon), -sin(lat)]])


# Rotate from b-frame to g-frame
def Cb_g(roll, pitch, yaw):
    return Rz(yaw)@Ry(pitch)@Rx(roll)


# Estimate roll and pitch from ZUPT (ned)
def acc2euler(ax, ay, az):
    roll = arctan2(ay, az)
    pitch = -arctan2(ax, sqrt(ay**2 + az**2))
    return roll, pitch


# Corrected atan2
def arctanc(y, x):
    z = arctan2(y, x)
    return (2*pi + z) % 2*pi


# Coordinate axis definitions
ned2enu = array([[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, -1]])

nwu2enu = array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 1]])

nwu2ned = array([[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, -1]])
