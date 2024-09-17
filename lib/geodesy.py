# Import libraries
from numpy import array, pi, sin, arcsin, cos, tan, arctan, sqrt
from convert import arctanc
from rotation import Ce_g


# Meridional radius of curvature
def Mrad(a, b, lat):
    """Calculate meridional radius of curvature."""
    e2 = (a**2 - b**2) / a**2
    M = a * (1 - e2) / (1 - e2 * sin(lat)**2)**(3/2)
    return M


# Normal radius of curvature
def Nrad(a, b, lat):
    """Calculate normal radius of curvature."""
    e2 = (a**2 - b**2) / a**2
    N = a / (1 - e2 * sin(lat)**2)**(1/2)
    return N


# Mean radius of curvature
def Rm(a, b, lat):
    """Calculate mean radius of curvature."""
    m = Mrad(a, b, lat)
    n = Nrad(a, b, lat)
    return sqrt(m * n)


# Radius of curvature for given azimuth (Euler's equation)
def Ra(a, b, lat, az):
    """Calculate radius of curvature for given azimuth (Euler's equation)."""
    M = Mrad(a, b, lat)
    N = Nrad(a, b, lat)
    return M * N / (M * sin(az)**2 + N * cos(az)**2)


# Meridional arc distance
def Marc(a, b, lat):
    """Calculate meridional arc distance."""
    f = (a - b) / a
    b0 = a * (1 - 1/2*f + 1/16*f**2 + 1/32*f**3)
    
    B = b0 * (lat - (3/4*f + 3/8*f**2 + 15/128*f**3) * sin(2*lat)
              + (15/64*f**2 + 15/64*f**3) * sin(4*lat)
              - 35/384*f**3 * sin(6*lat))
    return B


# Footpoint latitude
def footlat(a, b, x, lat0):
    """Calculate footpoint latitude."""
    f = (a - b) / a
    b0 = a * (1 - 1/2*f + 1/16*f**2 + 1/32*f**3)
    
    B = Marc(a, b, lat0) + x
    
    latf = B/b0 + (3/4*f + 3/8*f**2 + 21/256*f**3) * sin(2*B/b0) \
           + (21/64*f**2 + 21/64*f**3) * sin(4*B/b0) \
           + 151/768*f**3 * sin(6*B/b0)
    return latf


# Convert from ECEF coordinates to enu coordinates
def ECEF2enu(lat0, lon0, dP):
    """Convert from ECEF coordinates to ENU coordinates."""
    dP0 = Ce_g(lat0, lon0) @ dP
    
    # Convert from NED to ENU
    dP[0] = dP0[1]
    dP[1] = dP0[0]
    dP[2] = -dP0[2]
    return dP


# Convert from ECEF coordinates to ned coordinates
def ECEF2ned(lat0, lon0, dP):
    """Convert from ECEF coordinates to NED coordinates."""
    dP = Ce_g(lat0, lon0) @ dP
    return dP


# Convert geodetic coordinates to ECEF coordinates
def geod2ECEF(a, b, lat, lon, h):
    """Convert geodetic coordinates to ECEF coordinates."""
    N = Nrad(a, b, lat)
    
    P = array([[(N + h) * cos(lat) * cos(lon)],
               [(N + h) * cos(lat) * sin(lon)],
               [((b**2/a**2) * N + h) * sin(lat)]])
    return P


# Convert ECEF coordinates to geodetic coordinates (iteration)
def ECEF2geod(a, b, P):
    """Convert ECEF coordinates to geodetic coordinates (iteration)."""
    N = None
    X = P[0, 0]
    Y = P[1, 0]
    Z = P[2, 0]

    e2 = (a**2 - b**2) / a**2

    p = sqrt(X**2 + Y**2)
    lat_new = arctan(Z / p)

    epsilon = 1e-10
    lat = 0

    while abs(lat_new - lat) > epsilon:
        lat = lat_new
        N = Nrad(a, b, lat)
        lat_new = arctan(Z / p + N * e2 * sin(lat) / p)

    lat = lat_new
    lon = arctan(Y / X)
    h = p * cos(lat) + Z * sin(lat) - N * (1 - e2 * sin(lat)**2)

    return lat, lon, h


# Convert ECEF coordinates to geodetic coordinates (Vermeille, 2004)
def ECEF2geodv(a, b, P):
    """Convert ECEF coordinates to geodetic coordinates (Vermeille, 2004)."""
    X = P[0, 0]
    Y = P[1, 0]
    Z = P[2, 0]

    e2 = (a**2 - b**2) / a**2
    p = (X**2 + Y**2) / a**2
    q = (1 - e2) / a**2 * Z**2
    r = (p + q - e2**2) / 6
    s = e2**2 * (p * q) / (4 * r**3)
    t = (1 + s + sqrt(s * (2 + s)))**(1/3)
    u = r * (1 + t + 1/t)
    v = sqrt(u**2 + e2**2 * q)
    w = e2 * (u + v - q) / (2 * v)
    k = sqrt(u + v + w**2) - w
    D = k * sqrt(X**2 + Y**2) / (k + e2)

    lat = 2 * arctan(Z / (D + sqrt(D**2 + Z**2)))
    lon = arctan(Y/X)
    h = (k + e2 - 1)/k*sqrt(D**2 + Z**2)

    return lat, lon, h


# Geodetic direct problem
def geod1(a, b, lat1, lon1, az1, d):
    """Solve the geodetic direct problem."""
    f = (a - b)/a
    e2m = (a**2 - b**2)/b**2

    beta1 = arctan(b / a * tan(lat1))
    az0 = arcsin(sin(az1) * cos(beta1))
    sigma1 = arctan(tan(beta1) / cos(az1))

    g = e2m * cos(az0)**2
    H = 1/8*g - 1/16*g**2 + 37/1024*g**3
    b0 = b * (1 + 1/4*g - 3/64*g**2 + 5/256*g**3)

    d1 = b0 * (sigma1 - H * sin(2 * sigma1) - H**2 / 4 * sin(4 * sigma1) - H**3 / 6 * sin(6 * sigma1))
    d2 = d1 + d

    sigma2 = d2 / b0 + (H - 3/4*H**3)*sin(2*d2/b0) + 5/4*H**2*sin(4*d2/b0) + 29/12*H**3*sin(6*d2/b0)
    sigma = sigma2 - sigma1

    X = cos(beta1) * cos(sigma) - sin(beta1) * sin(sigma) * cos(az1)
    Y = sin(sigma)*sin(az1)
    Z = sin(beta1)*cos(sigma) + cos(beta1)*sin(sigma)*cos(az1)

    beta2 = arctan(Z/sqrt(X**2 + Y**2))
    dlon = arctan(Y/X)

    K = (f + f**2)/4*cos(az0)**2 - f**2/4*cos(az0)**4
    dlon = dlon - f * sin(az0) * ((1 - K - K**2) * sigma + K * sin(sigma) * cos(sigma1 + sigma2)
                                  + K**2 * sin(sigma) * cos(sigma) * cos(2 * (sigma1 + sigma2)))

    lat2 = arctan(a / b * tan(beta2))
    lon2 = lon1 + dlon
    az2 = arctanc(sin(az1) * cos(beta1), (cos(beta1) * cos(sigma) * cos(az1) - sin(beta1) * sin(sigma)))

    if az2 < pi:
        az2 = az2 + pi
    else:
        az2 = az2 - pi

    return lat2, lon2, az2


# Geodetic indirect problem
def geod2(a, b, lat1, lon1, lat2, lon2):
    """Solve the geodetic indirect problem."""
    f = (a - b)/a
    e2m = (a**2 - b**2)/b**2

    beta1 = arctan(b / a * tan(lat1))
    beta2 = arctan(b / a * tan(lat2))

    epsilon = 1e-10
    dlon_new = lon2 - lon1
    dlon = 0

    while abs(dlon_new - dlon) > epsilon:
        dlon = dlon_new

        X = cos(beta1) * sin(beta2) - sin(beta1) * cos(beta2) * cos(dlon)
        Y = cos(beta2)*sin(dlon)
        Z = sin(beta1)*sin(beta2) + cos(beta1)*cos(beta2)*cos(dlon)

        sigma = arctan(sqrt(X**2 + Y**2)/Z)
        az1 = arctanc(Y, X)
        az0 = arcsin(sin(az1)*cos(beta1))

        sigma1 = arctan(tan(beta1)/cos(az1))
        sigma2 = sigma1 + sigma

        K = (f + f**2)/4*cos(az0)**2 - f**2/4*cos(az0)**4

        dlon_new = (lon2 - lon1) + f * sin(az0) * ((1 - K - K**2) * sigma + K * sin(sigma) * cos(sigma1 + sigma2)
                                               + K**2*sin(sigma)*cos(sigma)*cos(2*(sigma1 + sigma2)))

    dlon = dlon_new
    az2 = arctanc(cos(beta1) * sin(dlon), (cos(beta1) * sin(beta2) * cos(dlon) - sin(beta1) * cos(beta2)))

    if az2 < pi:
        az2 = az2 + pi
    else:
        az2 = az2 - pi

    g = e2m*cos(az0)**2
    H = 1/8*g - 1/16*g**2 + 37/1024*g**3
    b0 = b*(1 + 1/4*g - 3/64*g**2 + 5/256*g**3)

    d = b0 * (sigma - 2 * H * sin(sigma) * cos(sigma1 + sigma2)
              - H**2 / 2 * sin(2 * sigma) * cos(2 * (sigma1 + sigma2))
              - H**3 / 3 * sin(3 * sigma) * cos(3 * (sigma1 + sigma2)))

    return az1, az2, d


# Convert geodetic coordinates to Transversal Mercator coordinates
def geod2TMgrid(a, b, lat, lon, lat0, lon0, scale, fnorth, feast):
    """Convert geodetic coordinates to Transversal Mercator coordinates."""
    B = Marc(a, b, lat) - Marc(a, b, lat0)
    N = Nrad(a, b, lat)
    e2 = (a**2 - b**2)/a**2
    eps2 = e2/(1 - e2)*cos(lat)**2
    dlon = lon - lon0

    x = B + 1/2 * dlon**2 * N * sin(lat) * cos(lat) \
        + 1/24 * dlon**4 * N * sin(lat) * cos(lat)**3 * (5 - tan(lat)**2 + 9*eps2 + 4*eps2**2) \
        + 1/720 * dlon**6 * N * sin(lat) * cos(lat)**5 * (61 - 58*tan(lat)**2 + tan(lat)**4)

    y = dlon * N * cos(lat) + 1/6 * dlon**3 * N * cos(lat)**3 * (1 - tan(lat)**2 + eps2) \
        + 1/120 * dlon**5 * N * cos(lat)**5 * (5 - 18 * tan(lat)**2 + tan(lat)**4)

    north = x * scale
    east = y * scale

    north = north + fnorth
    east = east + feast

    return north, east


# Convert Transversal Mercator coordinates to geodetic coordinates
def TMgrid2geod(a, b, north, east, lat0, lon0, scale, fnorth, feast):
    """Convert Transversal Mercator coordinates to geodetic coordinates."""
    north = north - fnorth
    east = east - feast

    x = north / scale
    y = east / scale

    latf = footlat(a, b, x, lat0)

    e2 = (a**2 - b**2) / a**2

    Mf = Mrad(a, b, latf)
    Nf = Nrad(a, b, latf)
    eps2f = e2 / (1 - e2) * cos(latf)**2

    lat = latf - 1/2 * y**2 * tan(latf) / (Mf * Nf) \
          + 1/24 * y**4 * tan(latf) / (Mf * Nf**3) * (5 + 3 * tan(latf)**2 + eps2f - 9 * eps2f * tan(latf)**2 - 4 * eps2f**2) \
          - 1/720 * y**6 * tan(latf) / (Mf * Nf**5) * (61 + 90 * tan(latf)**2 + 45 * tan(latf)**4)

    dlon = y / (Nf * cos(latf)) \
           - 1/6 * y**3 / (Nf**3 * cos(latf)) * (1 + 2 * tan(latf)**2 + eps2f) \
           + 1/120 * y**5 / (Nf**5 * cos(latf)) * (5 + 28 * tan(latf)**2 + 24 * tan(latf)**4)

    lon = dlon + lon0

    return lat, lon


# Transversal Mercator distance and azimuth correction
def TMcorr(a, b, x1, y1, x2, y2, lat0):
    """Calculate Transversal Mercator distance and azimuth correction."""
    latf = footlat(a, b, (x1 + x2) / 2, lat0)

    s = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    az = arctanc(y2 - y1, x2 - x1)
    R = Ra(a, b, latf, az)

    daz = -(x2 - x1) / (6 * R**2) * (2 * y1 + y2)
    ds = s / (6 * R**2) * (y1**2 + y1 * y2 + y2**2)

    return daz, ds


# Transversal Mercator meridian convergence
def TMconv(a, b, x, y, lat0):
    """Calculate Transversal Mercator meridian convergence."""
    latf = footlat(a, b, x, lat0)

    e2 = (a**2 - b**2) / a**2
    Nf = Nrad(a, b, latf)
    ef2 = e2 / (1 - e2) * cos(latf)**2

    gamma = y * tan(latf) / Nf \
            - y**3 * tan(latf) / (3 * Nf**3) * (1 + tan(latf)**2 - ef2 - 2 * ef2**2)

    return gamma


# Example
if __name__ == '__main__':

    # Import libraries
    from convert import deg2rad, rad2dms, dms2rad
    from rotation import Rx, Ry, Rz

    # Given coordinates EU89
    N_EU89 = 6615663.888  # meter
    E_EU89 = 600113.253   # meter
    h_EU89 = 156.228      # meter
    print(f"{N_EU89:.3f}m, {E_EU89:.3f}m, {h_EU89:.3f}m")

    # GRS80 ellipsoid
    a_GRS80 = 6378137
    f_GRS80 = 1/298.257222101
    b_GRS80 = a_GRS80*(1 - f_GRS80)

    # UTM projection
    lat0_EU89 = 0
    lon0_EU89 = deg2rad(9)  # zone 32V
    scale_EU89 = 0.9996
    fnorth_EU89 = 0
    feast_EU89 = 500000

    # Convert from projection to geodetic
    lat_EU89, lon_EU89 = TMgrid2geod(a_GRS80, b_GRS80, N_EU89, E_EU89,
                                     lat0_EU89, lon0_EU89, scale_EU89, fnorth_EU89, feast_EU89)
    print(rad2dms(lat_EU89), rad2dms(lon_EU89), f"{h_EU89:.3f}m")

    # Convert from geodetic to ECEF
    P_EU89 = geod2ECEF(a_GRS80, b_GRS80, lat_EU89, lon_EU89, h_EU89)

    # Parameters 7-parameter transformation (NMBU campus)
    T = array([[-313.368],
               [125.818],
               [-626.643]])

    m = (1 + 7.781959e-6)

    rx = dms2rad((0, 0, 2.336248))
    ry = dms2rad((0, 0, 1.712020))
    rz = dms2rad((0, 0, -1.169871))
    R = Rx(rx)@Ry(ry)@Rz(rz)

    P_NGO = T + m*R@P_EU89

    # Modified Bessel ellipsoid
    a_bess = 6377492.0176
    f_bess = 1/299.15281285
    b_bess = a_bess*(1 - f_bess)

    # TM projection
    lat0_NGO = deg2rad(58)
    lon0_NGO = dms2rad((10, 43, 22.5))  # axis 3
    scale_NGO = 1
    fnorth_NGO = 0
    feast_NGO = 0

    # Convert from ECEF to geodetic
    lat_NGO, lon_NGO, h_NGO = ECEF2geod(a_bess, b_bess, P_NGO)
    print(rad2dms(lat_NGO), rad2dms(lon_NGO), f"{h_NGO:.3f}m")

    # Convert from geodetic to projection (NGO48)
    x_NGO, y_NGO = geod2TMgrid(a_bess, b_bess, lat_NGO, lon_NGO,
                               lat0_NGO, lon0_NGO, scale_NGO, fnorth_NGO, feast_NGO)
    print(f"{x_NGO:.3f}m, {y_NGO:.3f}m, {h_NGO:.3f}m")
