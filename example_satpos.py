# Import libraries
from numpy import array
from numpy.linalg import norm
from lib.gnss import satpos
from lib.rotation import Rz
from lib.constants import c, OMEGADOTe


# Approximate receiver position [m]
Xr = array([[3172870.7170],
            [604208.2810],
            [5481574.2300]])

# Satellite G01 broadcast ephemerides (RINEX)
ttr = 8134                        # [s]
toe = 7200                        # [s]
ROOTa = 5.153634706497e+03        # [sqrt(m)]
DELTAn = 4.646979279625e-09       # [rad/s]
M0 = 9.760178388778e-01           # [rad]
e = 9.364774916321e-03            # [unitless]
omega = 7.546597134633e-01        # [rad]
Cus = 1.266598701477e-07          # [rad]
Cuc = -2.680346369743e-06         # [rad]
Crs = -5.456250000000e+01         # [m]
Crc = 3.865625000000e+02          # [m]
Cis = 1.285225152969e-07          # [rad]
Cic = -8.940696716309e-08         # [rad]
i0 = 9.785394956406e-01           # [rad]
iDOT = -3.500145795122e-10        # [rad/s]
OMEGA0 = -1.328259931335e+00      # [rad]
OMEGADOT = -8.668218208939e-09    # [rad/s]

# Satellite ECEF position @ 02:15:34 [m]
Xs0 = satpos(ttr, toe, ROOTa, DELTAn, M0, e, omega, Cus, Cuc, Crs, Crc, Cis, Cic, i0, iDOT, OMEGA0, OMEGADOT)
print(Xs0)

# Estimate of signal delay
sd_new = norm(Xs0 - Xr)/c
sd = 0

# Estimate signal travel time due to earth rotation
epsilon = 1e-10
Xs = None
while abs(sd_new - sd) > epsilon:
    sd = sd_new
    Xs = Rz(-OMEGADOTe*sd)@Xs0

    # Compute delay estimate
    sd_new = norm(Xs - Xr)/c

# Estimate of signal delay [ms]
print(norm(Xs-Xr)/c*1e3)

# Corrected satellite ECEF position @ 02:15:34 [m]
print(Xs)

# Change in satellite position due to earth rotation [m]
print(norm(Xs-Xs0))
