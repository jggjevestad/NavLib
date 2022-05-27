# Import libraries
from numpy import pi, arctan2, fmod, fix


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


# Convert from degree to degree, minutes, seconds
def deg2dms(deg):
    frac = abs(deg - int(deg))
    d = fix(deg)
    dmin = frac*60

    frac = abs(dmin - int(dmin))
    m = fix(dmin)
    s = frac*60
    return d, m, s


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
