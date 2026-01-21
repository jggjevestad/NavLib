# Import libraries
from numpy import array, eye, vstack, sin, cos, sqrt, arcsin, arctan2, pi, trace
from numpy.linalg import norm
from navlib.convert import deg2rad, rad2deg


# Directional Cosine Matrix (DCM)
# x-axis rotation
def Rx(rx):
    return array([[1, 0, 0],
                  [0, cos(rx), sin(rx)],
                  [0, -sin(rx), cos(rx)]])


# y-axis rotation
def Ry(ry):
    return array([[cos(ry), 0, -sin(ry)],
                  [0, 1, 0],
                  [sin(ry), 0, cos(ry)]])


# z-axis rotation
def Rz(rz):
    return array([[cos(rz), sin(rz), 0],
                  [-sin(rz), cos(rz), 0],
                  [0, 0, 1]])


# dcm2euler
def dcm2euler(C):
    roll = arctan2(-C[2, 1], C[2, 2])
    pitch = arctan2(C[2, 0], sqrt(C[2, 1]**2 + C[2, 2]**2))
    yaw = arctan2(-C[1, 0], C[0, 0])
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


# Coordinate axis definitions
uen2enu = array([[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]])

ned2enu = array([[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, -1]])

nwu2enu = array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 1]])

nwu2ned = array([[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, -1]])


# Example
def main():
    # Euler angles
    my_roll = deg2rad(10.0)
    my_pitch = deg2rad(20.0)
    my_yaw = deg2rad(30.0)

    # Euler to DCM
    my_C = euler2dcm(my_roll, my_pitch, my_yaw)
    print("euler2dcm:")
    print(my_C, norm(my_C, 2))

    # DCM to euler
    my_roll, my_pitch, my_yaw = dcm2euler(my_C)
    print("dcm2euler:")
    print(rad2deg(my_roll), rad2deg(my_pitch), rad2deg(my_yaw))

    # Euler to quaternion
    my_q = euler2quat(my_roll, my_pitch, my_yaw)
    print("euler2quat:")
    print(my_q, norm(my_q, 2))

    # Quaternion to euler
    my_roll, my_pitch, my_yaw = quat2euler(my_q)
    print("quat2euler:")
    print(rad2deg(my_roll), rad2deg(my_pitch), rad2deg(my_yaw))

    # DCM to axis_angle
    my_theta, my_r = dcm2axis_ang(my_C)
    print("DCM2axis_ang:")
    print(my_theta, my_r.T)

    # Axis_angle to DCM
    C1 = axis_ang2dcm(my_theta, my_r)
    print("axis_ang2DCM:")
    print(my_C, norm(my_C, 2))

    # DCM to quaternion
    my_q = dcm2quat(my_C)
    print("dcm2quat:")
    print(my_q, norm(my_q, 2))

    # Quaternion to DCM
    my_C = quat2dcm(my_q)
    print("quat2dcm:")
    print(my_C, norm(my_C, 2))

    # Rotation by quaternions (full example)

    # Rotation angle
    my_theta = deg2rad(30)

    # Rotation axis (normalized)
    my_r = array([[0],
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
    my_q = vstack([cos(my_theta/2),
                   sin(my_theta/2)*my_r])
    print("Rotation quaternion:")
    print(my_q, norm(my_q, 2))

    # Rotate p1 to p2
    p2q = qmult(qmult(my_q, p1q), qinv(my_q))
    print("Point p2:")
    print(p2q, norm(p2q, 2))

    # Corresponding dcm rotation

    # Quaternion to dcm
    my_C = quat2dcm(my_q)
    print("quat2dcm:")
    print(my_C, norm(my_C, 2))

    # Rotate p1 to p2
    p2 = my_C@p1
    print("Point p2:")
    print(p2, norm(p2, 2))


if __name__ == '__main__':
    main()
