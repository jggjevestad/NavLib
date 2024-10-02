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
def dms2deg(d: float, m: float, s: float) -> float:
    """Convert from degree, minutes, seconds to degree"""
    return abs(d) + m / 60 + s / 3600


# Convert from degree to degree, minutes, seconds
def deg2dms(deg: float) -> Tuple[int, int, float]:
    """Convert from degree to degree, minutes, seconds"""
    d = int(fix(deg))
    m = int(fix(abs(deg - d) * 60))
    s = (abs(deg - d) * 60 - m) * 60
    return d, m, s


# Convert from degree, minutes, seconds to radian
def dms2rad(d: float, m: float, s:float) -> float:
    """Convert from degree, minutes, seconds to radian"""
    return deg2rad(dms2deg(d, m, s))


# Convert from radian to degree, minutes, seconds
def rad2dms(rad: float) -> Tuple[int, int, float]:
    """Convert from radian to degree, minutes, seconds"""
    return deg2dms(rad2deg(rad))


# Example
def main():
    """Main function to demonstrate DMS to degrees conversion."""
    d, m, s = (59, 40, 1.10173)
    print(f"Original DMS: {d}° {m}' {s:.5f}\"")

    deg = dms2deg(d, m, s)
    print(f"Converted to degrees: {deg:.10f}")

    d, m, s = deg2dms(deg)
    print(f"Converted back to DMS: {d}° {m}' {s:.5f}\"")

if __name__ == '__main__':
    main()
