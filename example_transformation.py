# Import libraries
from numpy import array
from lib.convert import deg2rad, rad2dms, dms2rad
from lib.geodesy import geod2ECEF, ECEF2geod, TMgrid2geod, geod2TMgrid
from lib.rotation import Rx, Ry, Rz


# Given coordinates EU89
N = 6615663.888
E = 600113.253
h = 156.228
print(N, E, h)

# GRS80 ellipsoid
a = 6378137
f = 1/298.257222101
b = a*(1 - f)

# UTM projection
lat0 = 0
lon0 = deg2rad(9)  # zone 32V
scale = 0.9996
fnorth = 0
feast = 500000

# Convert from projection to geodetic
lat, lon = TMgrid2geod(a, b, N, E, lat0, lon0, scale, fnorth, feast)
print(rad2dms(lat), rad2dms(lon), h)

# Convert from geodetic to ECEF
P = geod2ECEF(a, b, lat, lon, h)

# Parameters 7-parameter transformation (NMBU campus)
T = array([[-313.368],
           [125.818],
           [-626.643]])

m = (1 + 7.781959e-6)

rx = dms2rad((0, 0, -2.336248))
ry = dms2rad((0, 0, -1.712020))
rz = dms2rad((0, 0, 1.169871))
R = Rx(rx)@Ry(ry)@Rz(rz)

P = T + m*R@P

# Modified Bessel ellipsoid
a = 6377492.0176
f = 1/299.15281285
b = a*(1 - f)

# TM projection
lat0 = deg2rad(58)
lon0 = dms2rad((10, 43, 22.5))  # axis 3
scale = 1
fnorth = 0
feast = 0

# Convert from ECEF to geodetic
lat, lon, h = ECEF2geod(a, b, P)
print(rad2dms(lat), rad2dms(lon), h)

# Convert from geodetic to projection (NGO48)
N, E = geod2TMgrid(a, b, lat, lon, lat0, lon0, scale, fnorth, feast)
print(N, E, h)
