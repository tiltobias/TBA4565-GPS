import georinex as gr
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")

ephimerides = gr.load("ephimerides.nav").to_dataframe()
obs = pd.read_csv("observations.csv")

satellites = [
    pd.concat([ephimerides.xs("G08", level="sv").iloc[0], obs[obs["sv"]=="G08"].iloc[0]]),
    pd.concat([ephimerides.xs("G10", level="sv").iloc[0], obs[obs["sv"]=="G10"].iloc[0]]),
    pd.concat([ephimerides.xs("G21", level="sv").iloc[0], obs[obs["sv"]=="G21"].iloc[0]]),
    pd.concat([ephimerides.xs("G24", level="sv").iloc[0], obs[obs["sv"]=="G24"].iloc[0]]),
    pd.concat([ephimerides.xs("G17", level="sv").iloc[1], obs[obs["sv"]=="G17"].iloc[0]]),
    pd.concat([ephimerides.xs("G03", level="sv").iloc[1], obs[obs["sv"]=="G03"].iloc[0]]),
    pd.concat([ephimerides.xs("G14", level="sv").iloc[0], obs[obs["sv"]=="G14"].iloc[0]]),
]

T = 558000
c = 299792458

GM = 3.986005e14
omega_e = 7.2921151467e-5
pi = 3.1415926535898

def R1(x): #rotation matrix around x axis
    return np.array([[1, 0, 0],
                     [0, np.cos(x), -np.sin(x)],
                     [0, np.sin(x), np.cos(x)]])

def R3(x): #rotation matrix around z axis
    return np.array([[np.cos(x), -np.sin(x), 0],
                     [np.sin(x), np.cos(x), 0],
                     [0, 0, 1]])

def satellite_coordinates(sat, T, correction=True):

    deltaN = sat["DeltaN"] if correction else 0
    Idot = sat["IDOT"] if correction else 0
    OmegaDot = sat["OmegaDot"] if correction else 0
    Cuc = sat["Cuc"] if correction else 0
    Cus = sat["Cus"] if correction else 0
    Crc = sat["Crc"] if correction else 0
    Crs = sat["Crs"] if correction else 0
    Cic = sat["Cic"] if correction else 0
    Cis = sat["Cis"] if correction else 0

    # print(sat)
    t_s = T - sat["P"]/c + sat["dt"]

    t_k = t_s - sat["Toe"]
    if t_k >  302400: t_k -= 604800
    if t_k < -302400: t_k += 604800

    M_k = sat["M0"] + (GM**0.5 / sat["sqrtA"]**3 + deltaN) * t_k

    E_k = M_k
    for _ in range(3):
        E_k = E_k + (M_k - E_k + sat["Eccentricity"] * np.sin(E_k)) / (1 - sat["Eccentricity"] * np.cos(E_k))

    f_k = 2 * np.arctan(np.sqrt((1 + sat["Eccentricity"]) / (1 - sat["Eccentricity"])) * np.tan(E_k / 2))

    u_k = sat["omega"] + f_k + Cuc * np.cos(2 * (sat["omega"] + f_k)) + Cus * np.sin(2 * (sat["omega"] + f_k))

    r_k = sat["sqrtA"]**2 * (1 - sat["Eccentricity"] * np.cos(E_k)) + Crc * np.cos(2 * (sat["omega"] + f_k)) + Crs * np.sin(2 * (sat["omega"] + f_k))

    i_k = sat["Io"] + Idot * t_k + Cic * np.cos(2 * (sat["omega"] + f_k)) + Cis * np.sin(2 * (sat["omega"] + f_k))

    lambda_k = sat["Omega0"] + (OmegaDot - omega_e) * t_k - omega_e * sat["Toe"]

    coords = R3(lambda_k) @ R1(i_k) @ R3(u_k) @ np.array([r_k, 0, 0]) # not negative rotation parameters because Y is flipped negative

    return coords

for sat in satellites:
    coords = satellite_coordinates(sat, T, correction=True)
    sat["coords"] = coords

print("\nCoordinates (ECEF) at transmission time:")
[print(sat["coords"]) for sat in satellites]


"""
Step 2:
"""

for sat in satellites:
    coords_uncorrected = satellite_coordinates(sat, T, correction=False)
    sat["coords_uncorrected"] = coords_uncorrected

print("\nUncorrected coordinates (ECEF) at transmission time:")
[print(sat["coords_uncorrected"]) for sat in satellites]

for sat in satellites:
    diff = sat["coords"] - sat["coords_uncorrected"]
    sat["diff"] = diff
    sat["diff_magnitude"] = np.linalg.norm(diff)

print("\nDifference between corrected and uncorrected coordinates:")
[print(sat["diff"], "magnitude:", sat["diff_magnitude"]) for sat in satellites]


"""
Step 3:
"""

approx_receiver_geodetic = np.array([63.2, 10.2, 100]) # lat, lon, height in degrees and meters

def geodetic_to_cartesian(geodetic_coords):
    """
    Convert geodetic coordinates (latitude, longitude, height) to ECEF Cartesian coordinates (X, Y, Z).
    Latitude and longitude are in degrees, height is in meters.
    Returns a numpy array [X, Y, Z] in meters.
    """
    phi, lam, h = np.deg2rad(geodetic_coords[0]), np.deg2rad(geodetic_coords[1]), geodetic_coords[2]
    a = 6378137.0 # WGS-84
    b = 6356752.3142

    N = a**2 / np.sqrt(a**2 * np.cos(phi)**2 + b**2 * np.sin(phi)**2)
    x = (N + h) * np.cos(phi) * np.cos(lam)
    y = (N + h) * np.cos(phi) * np.sin(lam)
    z = (b**2 / a**2 * N + h) * np.sin(phi)
    return np.array([x, y, z])


approx_receiver_cartesian = geodetic_to_cartesian(approx_receiver_geodetic)
print("\nApprox receiver coordinates in Cartesian:")
print(approx_receiver_cartesian)
