# NavLib


# Import libraries
from numpy import array, eye, vstack, sin, cos, tan, sqrt, arcsin, arctan, arctan2, pi, trace, fmod, fix
from numpy.linalg import norm


# Modified arctanc (returns quadrant independent angle, e.g. azimuth)
def arctanc(y, x):
    z = arctan2(y, x)
    return fmod(2*pi + z, 2*pi)


# Convert from degree to radian
def deg2rad(deg):
    return deg*(pi/180)


# Convert from radian to degree
def rad2deg(rad):
    return rad*(180/pi)


# Convert from gradian to radian
def grad2rad(grad):
    return grad*(pi/200)


# Convert from radian to gradian
def rad2grad(rad):
    return rad*(200/pi)


# Convert from semicircle to radian
def sc2rad(sc):
    return sc*pi


# Convert from radian to semicircle
def rad2sc(rad):
    return rad/pi


# Convert from degree, minutes, seconds to degree
def dms2deg(dms):
    d = dms[0]
    m = dms[1]
    s = dms[2]

    deg = abs(d) + m/60 + s/3600
    return deg


# Convert from degree, minutes, seconds to degree
def dm2deg(dm):
    d = dm[0]
    m = dm[1]

    deg = abs(d) + m/60
    return deg


# Convert from degree to degree, minutes, seconds
def deg2dms(deg):
    frac = abs(deg - int(deg))
    d = int(fix(deg))
    dmin = frac*60

    frac = abs(dmin - int(dmin))
    m = int(fix(dmin))
    s = frac*60
    return d, m, s


# Convert from degree to degree, minutes, seconds
def deg2dm(deg):
    frac = abs(deg - int(deg))
    d = int(fix(deg))
    m = frac*60

    return d, m


# Convert from degree, minutes, seconds to radian
def dms2rad(dms):
    deg = dms2deg(dms)
    rad = deg2rad(deg)
    return rad


# Convert from radian to degree, minutes, seconds
def rad2dms(rad):
    deg = rad2deg(rad)
    dms = deg2dms(deg)
    return dms


# Meridional radius of curvature
def Mrad(a, b, lat):
    e2 = (a**2 - b**2)/a**2
    M = a*(1 - e2)/(1 - e2*sin(lat)**2)**(3/2)
    return M


# Normal radius of curvature
def Nrad(a, b, lat):
    e2 = (a**2 - b**2)/a**2
    N = a/(1 - e2*sin(lat)**2)**(1/2)
    return N


# Mean radius of curvature
def Rm(a, b, lat):
    M = Mrad(a, b, lat)
    N = Nrad(a, b, lat)

    return sqrt(M*N)


# Radius of curvature for given azimuth (Euler's equation)
def Ra(a, b, lat, az):
    M = Mrad(a, b, lat)
    N = Nrad(a, b, lat)

    return M*N/(M*sin(az)**2 + N*cos(az)**2)


# Meridional arc distance
def Marc(a, b, lat):
    f = (a - b)/a
    b0 = a*(1 - 1/2*f + 1/16*f**2 + 1/32*f**3)

    B = b0*(lat - (3/4*f + 3/8*f**2 + 15/128*f**3)*sin(2*lat)
            + (15/64*f**2 + 15/64*f**3)*sin(4*lat)
            - 35/384*f**3*sin(6*lat))
    return B


# Footpoint latitude
def footlat(a, b, x, lat0):
    f = (a - b)/a
    b0 = a*(1 - 1/2*f + 1/16*f**2 + 1/32*f**3)

    B = Marc(a, b, lat0) + x

    latf = B/b0 + (3/4*f + 3/8*f**2 + 21/256*f**3)*sin(2*B/b0) \
           + (21/64*f**2 + 21/64*f**3)*sin(4*B/b0) \
           + 151/768*f**3*sin(6*B/b0)
    return latf


# Convert from ECEF coordinates to enu coordinates
def ECEF2enu(lat0, lon0, dP):
    dP0 = Ce_g(lat0, lon0)@dP

    # Convert from ned to enu
    dP[0] = dP0[1]
    dP[1] = dP0[0]
    dP[2] = -dP0[2]
    return dP


# Convert from ECEF coordinates to ned coordinates
def ECEF2ned(lat0, lon0, dP):
    dP = Ce_g(lat0, lon0)@dP
    return dP


# Convert geodetic coordinates to ECEF coordinates
def geod2ECEF(a, b, lat, lon, h):
    N = Nrad(a, b, lat)

    P = array([[(N + h)*cos(lat)*cos(lon)],
               [(N + h)*cos(lat)*sin(lon)],
               [((b**2/a**2)*N + h)*sin(lat)]])
    return P


# Convert ECEF coordinates to geodetic coordinates (iteration)
def ECEF2geod(a, b, P):
    N = None
    X = P[0, 0]
    Y = P[1, 0]
    Z = P[2, 0]

    e2 = (a**2 - b**2)/a**2

    p = sqrt(X**2 + Y**2)
    lat_new = arctan(Z/p)

    epsilon = 1e-10
    lat = 0

    while abs(lat_new - lat) > epsilon:
        lat = lat_new
        N = Nrad(a, b, lat)
        lat_new = arctan(Z/p + N*e2*sin(lat)/p)

    lat = lat_new
    lon = arctan(Y/X)
    h = p*cos(lat) + Z*sin(lat) - N*(1 - e2*sin(lat)**2)

    return lat, lon, h


# Convert ECEF coordinates to geodetic coordinates (Vermeille, 2004)
def ECEF2geodv(a, b, P):
    X = P[0, 0]
    Y = P[1, 0]
    Z = P[2, 0]

    e2 = (a**2 - b**2)/a**2
    p = (X**2 + Y**2)/a**2
    q = (1 - e2)/a**2*Z**2
    r = (p + q - e2**2)/6
    s = e2**2*(p*q)/(4*r**3)
    t = (1 + s + sqrt(s*(2 + s)))**(1/3)
    u = r*(1 + t + 1/t)
    v = sqrt(u**2 + e2**2*q)
    w = e2*(u + v - q)/(2*v)
    k = sqrt(u + v + w**2) - w
    D = k*sqrt(X**2 + Y**2)/(k + e2)

    lat = 2*arctan(Z/(D + sqrt(D**2 + Z**2)))
    lon = arctan(Y/X)
    h = (k + e2 - 1)/k*sqrt(D**2 + Z**2)

    return lat, lon, h


# Geodetic direct problem
def geod1(a, b, lat1, lon1, az1, d):
    f = (a - b)/a
    e2m = (a**2 - b**2)/b**2

    beta1 = arctan(b/a*tan(lat1))
    az0 = arcsin(sin(az1)*cos(beta1))
    sigma1 = arctan(tan(beta1)/cos(az1))

    g = e2m*cos(az0)**2
    H = 1/8*g - 1/16*g**2 + 37/1024*g**3
    b0 = b*(1 + 1/4*g - 3/64*g**2 + 5/256*g**3)

    d1 = b0*(sigma1 - H*sin(2*sigma1) - H**2/4*sin(4*sigma1) - H**3/6*sin(6*sigma1))
    d2 = d1 + d

    sigma2 = d2/b0 + (H - 3/4*H**3)*sin(2*d2/b0) + 5/4*H**2*sin(
        4*d2/b0) + 29/12*H**3*sin(6*d2/b0)
    sigma = sigma2 - sigma1

    X = cos(beta1)*cos(sigma) - sin(beta1)*sin(sigma)*cos(az1)
    Y = sin(sigma)*sin(az1)
    Z = sin(beta1)*cos(sigma) + cos(beta1)*sin(sigma)*cos(az1)

    beta2 = arctan(Z/sqrt(X**2 + Y**2))
    dlon = arctan(Y/X)

    K = (f + f**2)/4*cos(az0)**2 - f**2/4*cos(az0)**4
    dlon = dlon - f*sin(az0)*((1 - K - K**2)*sigma + K*sin(sigma)*cos(sigma1 + sigma2)
                              + K**2*sin(sigma)*cos(sigma)*cos(2*(sigma1 + sigma2)))

    lat2 = arctan(a/b*tan(beta2))
    lon2 = lon1 + dlon
    az2 = arctanc(sin(az1)*cos(beta1), (cos(beta1)*cos(sigma)*cos(az1) - sin(beta1)*sin(sigma)))

    if az2 < pi:
        az2 = az2 + pi
    else:
        az2 = az2 - pi

    return lat2, lon2, az2


# Geodetic indirect problem
def geod2(a, b, lat1, lon1, lat2, lon2):

    f = (a - b)/a
    e2m = (a**2 - b**2)/b**2

    beta1 = arctan(b/a*tan(lat1))
    beta2 = arctan(b/a*tan(lat2))

    epsilon = 1e-10
    dlon_new = lon2 - lon1
    dlon = 0

    while abs(dlon_new - dlon) > epsilon:
        dlon = dlon_new

        X = cos(beta1)*sin(beta2) - sin(beta1)*cos(beta2)*cos(dlon)
        Y = cos(beta2)*sin(dlon)
        Z = sin(beta1)*sin(beta2) + cos(beta1)*cos(beta2)*cos(dlon)

        sigma = arctan(sqrt(X**2 + Y**2)/Z)
        az1 = arctanc(Y, X)
        az0 = arcsin(sin(az1)*cos(beta1))

        sigma1 = arctan(tan(beta1)/cos(az1))
        sigma2 = sigma1 + sigma

        K = (f + f**2)/4*cos(az0)**2 - f**2/4*cos(az0)**4

        dlon_new = (lon2 - lon1) + f*sin(az0)*((1 - K - K**2)*sigma + K*sin(sigma)*cos(sigma1 + sigma2)
                                               + K**2*sin(sigma)*cos(sigma)*cos(2*(sigma1 + sigma2)))

    dlon = dlon_new
    az2 = arctanc(cos(beta1)*sin(dlon), (cos(beta1)*sin(beta2)*cos(dlon) - sin(beta1)*cos(beta2)))

    if az2 < pi:
        az2 = az2 + pi
    else:
        az2 = az2 - pi

    g = e2m*cos(az0)**2
    H = 1/8*g - 1/16*g**2 + 37/1024*g**3
    b0 = b*(1 + 1/4*g - 3/64*g**2 + 5/256*g**3)

    d = b0*(sigma - 2*H*sin(sigma)*cos(sigma1 + sigma2)
            - H**2/2*sin(2*sigma)*cos(2*(sigma1 + sigma2))
            - H**3/3*sin(3*sigma)*cos(3*(sigma1 + sigma2)))

    return az1, az2, d


# Convert geodetic coordinates to Transversal Mercator coordinates
def geod2TMgrid(a, b, lat, lon, lat0, lon0, scale, fnorth, feast):
    B = Marc(a, b, lat) - Marc(a, b, lat0)
    N = Nrad(a, b, lat)
    e2 = (a**2 - b**2)/a**2
    eps2 = e2/(1 - e2)*cos(lat)**2
    dlon = lon - lon0

    x = B + 1/2*dlon**2*N*sin(lat)*cos(lat) \
        + 1/24*dlon**4*N*sin(lat)*cos(lat)**3*(5 - tan(lat)**2 + 9*eps2 + 4*eps2**2) \
        + 1/720*dlon**6*N*sin(lat)*cos(lat)**5*(61 - 58*tan(lat)**2 + tan(lat)**4)

    y = dlon*N*cos(lat) + 1/6*dlon**3*N*cos(lat)**3*(1 - tan(lat)**2 + eps2) \
        + 1/120*dlon**5*N*cos(lat)**5*(5 - 18*tan(lat)**2 + tan(lat)**4)

    north = x*scale
    east = y*scale

    north = north + fnorth
    east = east + feast

    return north, east


# Convert Transversal Mercator coordinates to geodetic coordinates
def TMgrid2geod(a, b, north, east, lat0, lon0, scale, fnorth, feast):
    north = north - fnorth
    east = east - feast

    x = north/scale
    y = east/scale

    latf = footlat(a, b, x, lat0)

    e2 = (a**2 - b**2)/a**2

    Mf = Mrad(a, b, latf)
    Nf = Nrad(a, b, latf)
    eps2f = e2/(1 - e2)*cos(latf)**2

    lat = latf - 1/2*y**2*tan(latf)/(Mf*Nf) \
          + 1/24*y**4*tan(latf)/(Mf*Nf**3)*(5 + 3*tan(latf)**2 + eps2f - 9*eps2f*tan(latf)**2 - 4*eps2f**2) \
          - 1/720*y**6*tan(latf)/(Mf*Nf**5)*(61 + 90*tan(latf)**2 + 45*tan(latf)**4)

    dlon = y/(Nf*cos(latf)) \
           - 1/6*y**3/(Nf**3*cos(latf))*(1 + 2*tan(latf)**2 + eps2f) \
           + 1/120*y**5/(Nf**5*cos(latf))*(5 + 28*tan(latf)**2 + 24*tan(latf)**4)

    lon = dlon + lon0

    return lat, lon


# Transversal Mercator distance and azimuth correction
def TMcorr(a, b, x1, y1, x2, y2, lat0):
    latf = footlat(a, b, (x1 + x2)/2, lat0)

    s = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    az = arctanc(y2 - y1, x2 - x1)
    R = Ra(a, b, latf, az)

    daz = -(x2 - x1)/(6*R**2)*(2*y1 + y2)
    ds = s/(6*R**2)*(y1**2 + y1*y2 + y2**2)

    return daz, ds


# Transversal Mercator meridian convergence
def TMconv(a, b, x, y, lat0):
    latf = footlat(a, b, x, lat0)

    e2 = (a**2 - b**2)/a**2
    Nf = Nrad(a, b, latf)
    ef2 = e2/(1 - e2)*cos(latf)**2

    gamma = y*tan(latf)/Nf \
            -y**3*tan(latf)/(3*Nf**3)*(1 + tan(latf)**2 - ef2 - 2*ef2**2)

    return gamma


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