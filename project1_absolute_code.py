import georinex as gr
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")

from data_project1_absolute_code.variables import T

"""
General constants
"""
c = 299792458
GM = 3.986005e14
omega_e = 7.2921151467e-5
pi = 3.1415926535898


"""
Functions
"""

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


def A_matrix(satellites, approx_receiver_cartesian):
    A = []
    for sat in satellites:
        rho_i = np.linalg.norm(sat["coords"] - approx_receiver_cartesian[:3])
        row = [
            -(sat["coords"][0] - approx_receiver_cartesian[0]) / rho_i,
            -(sat["coords"][1] - approx_receiver_cartesian[1]) / rho_i,
            -(sat["coords"][2] - approx_receiver_cartesian[2]) / rho_i,
            -c
        ]
        A.append(row)
    return np.array(A)

def delta_L_vector(satellites, approx_receiver_cartesian):
    L = []
    for sat in satellites:
        rho_i = np.linalg.norm(sat["coords"] - approx_receiver_cartesian)
        L.append(sat["P"] - rho_i - c * sat["dt"] - sat["dion"] - sat["dtrop"])
    return np.array(L).reshape(-1, 1)


def cartesian_to_geodetic(cartesian_coords):
    """
    Convert ECEF Cartesian coordinates (X, Y, Z) to geodetic coordinates (latitude, longitude, height).
    X, Y, Z are in meters.
    Returns a numpy array [latitude, longitude, height] where latitude and longitude are in degrees and height is in meters.
    """
    x, y, z = cartesian_coords
    a = 6378137.0 # WGS-84
    b = 6356752.3142
    e2 = 1 - (b**2 / a**2)
    lam = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    phi_0 = np.arctan2(z, p * (1 - e2))
    for i in range(10):
        N_0 = a**2 / np.sqrt(a**2 * np.cos(phi_0)**2 + b**2 * np.sin(phi_0)**2)
        h = p / np.cos(phi_0) - N_0
        phi = np.arctan2(z, p * (1 - e2 * (N_0 / (N_0 + h))))
        if abs(phi - phi_0) < 1e-12: break
        if i == 9: 
            print("Max iterations reached")
            break
        phi_0 = phi
    return np.array([np.rad2deg(phi), np.rad2deg(lam), h])


"""
Steps
"""

def main():
    ephimerides = gr.load("data_project1_absolute_code/ephimerides.nav").to_dataframe()
    obs = pd.read_csv("data_project1_absolute_code/observations.csv")

    satellites = [
        pd.concat([ephimerides.xs("G08", level="sv").iloc[0], obs[obs["sv"]=="G08"].iloc[0]]),
        pd.concat([ephimerides.xs("G10", level="sv").iloc[0], obs[obs["sv"]=="G10"].iloc[0]]),
        pd.concat([ephimerides.xs("G21", level="sv").iloc[0], obs[obs["sv"]=="G21"].iloc[0]]),
        pd.concat([ephimerides.xs("G24", level="sv").iloc[0], obs[obs["sv"]=="G24"].iloc[0]]),
        pd.concat([ephimerides.xs("G17", level="sv").iloc[1], obs[obs["sv"]=="G17"].iloc[0]]),
        pd.concat([ephimerides.xs("G03", level="sv").iloc[1], obs[obs["sv"]=="G03"].iloc[0]]),
        pd.concat([ephimerides.xs("G14", level="sv").iloc[0], obs[obs["sv"]=="G14"].iloc[0]]),
    ]

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
    
    approx_receiver_cartesian = geodetic_to_cartesian(approx_receiver_geodetic)
    print("\nApprox receiver coordinates in Cartesian:")
    print(approx_receiver_cartesian)


    """
    Step 4:
    """
    
    receiver_cartesian = approx_receiver_cartesian.copy()
    receiver_clock_bias = 0
    for i in range(10):
        A = A_matrix(satellites, receiver_cartesian)
        delta_L = delta_L_vector(satellites, receiver_cartesian)
        delta_X = np.linalg.inv(A.T @ A) @ A.T @ delta_L
        receiver_cartesian += delta_X[:3].flatten()
        receiver_clock_bias = delta_X[3, 0]
        if np.linalg.norm(delta_X[:3]) < 1e-6: break
        if i == 9: 
            print("Max iterations reached")
            break
        print(f"Iteration {i+1}:", receiver_cartesian)

    print("\nFinal receiver coordinates in Cartesian:", receiver_cartesian)


    """
    Step 5:
    """

    Q_x = np.linalg.inv(A.T @ A)

    print("\nCovariance matrix Q_x:\n", Q_x)

    PDOP = np.sqrt(Q_x[0,0] + Q_x[1,1] + Q_x[2,2])
    print("PDOP:", PDOP)


    """
    Step 6:
    """
    
    receiver_geodetic = cartesian_to_geodetic(receiver_cartesian)
    print("\nFinal receiver coordinates in Geodetic (lat, lon, height):")
    print(receiver_geodetic)


    """
    Step 7:
    """

    print("\nReceiver clock bias (in seconds):", receiver_clock_bias)

if __name__ == "__main__":
    main()