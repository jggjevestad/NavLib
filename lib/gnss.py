# Import libraries
from numpy import array, sqrt, sin, cos, arctan2
from lib.constants import GM, OMEGADOTe


# Correction for beginning or end of week crossovers in GNSS systems
def dt(t, t0):
    t = t - t0

    if t > 302400:
        t = t - 604800
    elif t < -302400:
        t = t + 604800

    return t


# Satellite ECEF position
def satpos(ttr, toe, ROOTa, DELTAn, M0, e, omega, Cus, Cuc, Crs, Crc, Cis, Cic, i0, iDOT, OMEGA0, OMEGADOT):
    # Anomalies of the Keplerian orbit
    a = ROOTa**2        # Semi-major axis [m]
    n0 = sqrt(GM/a**3)  # Mean angular velocity [rad/sec]
    t = dt(ttr, toe)    # Time from reference epoch [s]
    n = n0 + DELTAn     # Corrected mean motion [rad/s]
    M = M0 + n*t        # Mean anomaly [rad]

    # Kepler's equation
    epsilon = 1e-10
    E_new = M
    E = 0

    while abs(E_new - E) > epsilon:
        E = E_new
        E_new = M + e*sin(E)

    # Eccentric anomaly
    E = E_new

    # True anomaly
    v = arctan2(sqrt(1 - e**2)*sin(E), cos(E) - e)

    # Argument of latitude
    PHI = v + omega

    # Second harmonic pertubations
    du = Cus*sin(2*PHI) + Cuc*cos(2*PHI)  # Argument of latitude correction [rad]
    dr = Crs*sin(2*PHI) + Crc*cos(2*PHI)  # Radius correction [m]
    di = Cis*sin(2*PHI) + Cic*cos(2*PHI)  # Inclination correction[rad]

    # Orbit corrections
    u = PHI + du               # Corrected argument of latitude [rad]
    r = a*(1 - e*cos(E)) + dr  # Corrected radius [m]
    i = i0 + di + iDOT*t       # Corrected inclination [rad]

    # Corrected longitude of ascending node
    OMEGA = OMEGA0 + (OMEGADOT - OMEGADOTe)*t - OMEGADOTe*toe

    # Satellite position in ECEF system
    Xs0 = array([[r*cos(u)*cos(OMEGA) - r*sin(u)*sin(OMEGA)*cos(i)],
                 [r*cos(u)*sin(OMEGA) + r*sin(u)*cos(OMEGA)*cos(i)],
                 [r*sin(u)*sin(i)]])

    return Xs0
