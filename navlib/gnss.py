# Import libraries
from numpy import array, sqrt, sin, cos, arctan2
from constants import GM, OMEGADOTe


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
    a = ROOTa**2          # Semi-major axis [m]
    n0 = sqrt(GM / a**3)  # Mean angular velocity [rad/sec]
    t = dt(ttr, toe)      # Time from reference epoch [s]
    n = n0 + DELTAn       # Corrected mean motion [rad/s]
    M = M0 + n * t        # Mean anomaly [rad]

    # Kepler's equation
    epsilon = 1e-10
    E_new = M
    E = 0

    while abs(E_new - E) > epsilon:
        E = E_new
        E_new = M + e * sin(E)

    # Eccentric anomaly
    E = E_new

    # True anomaly
    v = arctan2(sqrt(1 - e**2) * sin(E), cos(E) - e)

    # Argument of latitude
    PHI = v + omega

    # Second harmonic perturbations
    du = Cus * sin(2 * PHI) + Cuc * cos(2 * PHI)  # Argument of latitude correction [rad]
    dr = Crs * sin(2 * PHI) + Crc * cos(2 * PHI)  # Radius correction [m]
    di = Cis * sin(2 * PHI) + Cic * cos(2 * PHI)  # Inclination correction[rad]

    # Orbit corrections
    u = PHI + du                   # Corrected argument of latitude [rad]
    r = a * (1 - e * cos(E)) + dr  # Corrected radius [m]
    i = i0 + di + iDOT * t         # Corrected inclination [rad]

    # Corrected longitude of ascending node
    OMEGA = OMEGA0 + (OMEGADOT - OMEGADOTe) * t - OMEGADOTe * toe

    # Satellite position in ECEF system
    Xs0 = array([[r * cos(u) * cos(OMEGA) - r * sin(u) * sin(OMEGA) * cos(i)],
                 [r * cos(u) * sin(OMEGA) + r * sin(u) * cos(OMEGA) * cos(i)],
                 [r * sin(u) * sin(i)]])

    return Xs0


def main():
    # Import libraries
    from numpy.linalg import norm
    from constants import c
    from rotation import Rz

    # Approximate receiver position [m]
    Xr = array([[3172870.7170],
                [604208.2810],
                [5481574.2300]])

    # Satellite G01 broadcast ephemerides (RINEX)
    ttr = 8134                      # [s]
    toe = 7200                      # [s]
    ROOTa = 5.153634706497e+03      # [sqrt(m)]
    DELTAn = 4.646979279625e-09     # [rad/s]
    M0 = 9.760178388778e-01         # [rad]
    e = 9.364774916321e-03          # [unitless]
    omega = 7.546597134633e-01      # [rad]
    Cus = 1.266598701477e-07        # [rad]
    Cuc = -2.680346369743e-06       # [rad]
    Crs = -5.456250000000e+01       # [m]
    Crc = 3.865625000000e+02        # [m]
    Cis = 1.285225152969e-07        # [rad]
    Cic = -8.940696716309e-08       # [rad]
    i0 = 9.785394956406e-01         # [rad]
    iDOT = -3.500145795122e-10      # [rad/s]
    OMEGA0 = -1.328259931335e+00    # [rad]
    OMEGADOT = -8.668218208939e-09  # [rad/s]

    # Satellite ECEF position @ 02:15:34 [m]
    Xs0 = satpos(ttr, toe, ROOTa, DELTAn, M0, e, omega,
                 Cus, Cuc, Crs, Crc, Cis, Cic, i0, iDOT, OMEGA0, OMEGADOT)
    print(Xs0)

    # Estimate of signal delay
    sd_new = norm(Xs0 - Xr) / c
    sd = 0

    # Estimate signal travel time due to earth rotation
    Xs = Xs0
    while abs(sd_new - sd) > 1e-10:
        sd = sd_new
        Xs = Rz(-OMEGADOTe * sd) @ Xs0

        # Compute delay estimate
        sd_new = norm(Xs - Xr) / c

    # Estimate of signal delay [ms]
    print(f"Estimated signal delay: {norm(Xs - Xr) / c * 1e3:.3f} ms")

    # Corrected satellite ECEF position @ 02:15:34 [m]
    print(f"Corrected satellite ECEF position: {Xs[0][0]:.3f} m, {Xs[1][0]:.3f} m, {Xs[2][0]:.3f} m")

    # Change in satellite position due to earth rotation [m]
    print(f"Change in satellite position due to earth rotation: {norm(Xs - Xs0):.3f} m")


# Example
if __name__ == '__main__':
    main()
