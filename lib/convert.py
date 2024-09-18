# Import libraries
from typing import Tuple
from numpy import pi, arctan2, fix


# Modified arctanc (returns quadrant independent angle, e.g. azimuth)
def arctanc(y: float, x: float) -> float:
    """
    Modified arctanc (returns quadrant independent angle, e.g. azimuth)
    """
    return (arctan2(y, x) + 2*pi) % (2*pi)


# Convert from degree to radian
def deg2rad(deg: float) -> float:
    """Convert from degree to radian"""
    return deg * (pi / 180)


# Convert from radian to degree
def rad2deg(rad: float) -> float:
    """Convert from radian to degree"""
    return rad * (180 / pi)


# Convert from gradian to radian
def grad2rad(grad: float) -> float:
    """Convert from gradian to radian"""
    return grad * (pi / 200)


# Convert from radian to gradian
def rad2grad(rad: float) -> float:
    """Convert from radian to gradian"""
    return rad * (200 / pi)


# Convert from semicircle to radian
def sc2rad(sc: float) -> float:
    """Convert from semicircle to radian"""
    return sc * pi


# Convert from radian to semicircle
def rad2sc(rad: float) -> float:
    """Convert from radian to semicircle"""
    return rad / pi


# Convert from degree, minutes, seconds to degree
def dms2deg(dms: Tuple[float, float, float]) -> float:
    """Convert from degree, minutes, seconds to degree"""
    d, m, s = dms
    return abs(d) + m / 60 + s / 3600


# Convert from degree to degree, minutes, seconds
def deg2dms(deg: float) -> Tuple[int, int, float]:
    """Convert from degree to degree, minutes, seconds"""
    frac = abs(deg - int(deg))
    d = int(fix(deg))
    dmin = frac * 60

    frac = abs(dmin - int(dmin))
    m = int(fix(dmin))
    s = frac * 60
    return d, m, s


# Convert from degree, minutes, seconds to radian
def dms2rad(dms: Tuple[float, float, float]) -> float:
    """Convert from degree, minutes, seconds to radian"""
    return deg2rad(dms2deg(dms))


# Convert from radian to degree, minutes, seconds
def rad2dms(rad: float) -> Tuple[int, int, float]:
    """Convert from radian to degree, minutes, seconds"""
    return deg2dms(rad2deg(rad))


# Example
def main():
    dms = (59, 40, 1.10173)
    print(f"Original DMS: {dms}")

    deg = dms2deg(dms)
    print(f"Converted to degrees: {deg:.10f}")

    dms_result = deg2dms(deg)
    print(f"Converted back to DMS: {dms_result}")

if __name__ == '__main__':
    main()
