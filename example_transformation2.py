# Import libraries
from numpy import array
from lib.convert import deg2rad, rad2dms
from lib.geodesy import ECEF2geod, geod2TMgrid


# Given coordinates AASC ECEF
P_AASC = array([[3172870.7158],
                [604208.2821],
                [5481574.2310]])

P_FLO3 = array([[3172409.8850],
                [603897.9350],
                [5481884.6911]])

P_OPEC = array([[3149785.9591],
                [598260.8780],
                [5495348.4875]])

P_OSLS = array([[3169982.1455],
                [579956.5934],
                [5485936.4657]])

# GRS80 ellipsoid
a_GRS80 = 6378137
f_GRS80 = 1/298.257222101
b_GRS80 = a_GRS80*(1 - f_GRS80)
print(b_GRS80)

# WGS84 ellipsoid
a_WGS84 = 6378137
f_WGS84 = 1/298.257223563
b_WGS84 = a_WGS84*(1 - f_WGS84)
print(b_WGS84)

# UTM projection
lat0 = 0
lon0 = deg2rad(9)  # zone 32V
scale = 0.9996
fnorth = 0
feast = 500000

# Convert from ECEF to geodetic
lat_AASC, lon_AASC, h_AASC = ECEF2geod(a_GRS80, b_GRS80, P_AASC)
print(rad2dms(lat_AASC), rad2dms(lon_AASC), h_AASC)

lat_FLO3, lon_FLO3, h_FLO3 = ECEF2geod(a_GRS80, b_GRS80, P_FLO3)
print(rad2dms(lat_FLO3), rad2dms(lon_FLO3), h_FLO3)

lat_OPEC, lon_OPEC, h_OPEC = ECEF2geod(a_GRS80, b_GRS80, P_OPEC)
print(rad2dms(lat_OPEC), rad2dms(lon_OPEC), h_OPEC)

lat_OSLS, lon_OSLS, h_OSLS = ECEF2geod(a_GRS80, b_GRS80, P_OSLS)
print(rad2dms(lat_OSLS), rad2dms(lon_OSLS), h_OSLS)

# Convert from geodetic to projection (EU89)
N_AASC, E_AASC = geod2TMgrid(a_GRS80, b_GRS80, lat_AASC, lon_AASC, lat0, lon0, scale, fnorth, feast)
print(N_AASC, E_AASC, h_AASC)

N_FLO3, E_FLO3 = geod2TMgrid(a_GRS80, b_GRS80, lat_FLO3, lon_FLO3, lat0, lon0, scale, fnorth, feast)
print(N_FLO3, E_FLO3, h_FLO3)

N_OPEC, E_OPEC = geod2TMgrid(a_GRS80, b_GRS80, lat_OPEC, lon_OPEC, lat0, lon0, scale, fnorth, feast)
print(N_OPEC, E_OPEC, h_OPEC)

N_OSLS, E_OSLS = geod2TMgrid(a_GRS80, b_GRS80, lat_OSLS, lon_OSLS, lat0, lon0, scale, fnorth, feast)
print(N_OSLS, E_OSLS, h_OSLS)
