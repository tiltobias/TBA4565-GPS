import georinex as gr
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")

data = gr.load("ephimerides.nav").to_dataframe()
obs = pd.read_csv("observations.csv")

satellites = [
    pd.concat([data.xs("G08", level="sv").iloc[0], obs[obs["sv"]=="G08"].iloc[0]]),
    pd.concat([data.xs("G10", level="sv").iloc[0], obs[obs["sv"]=="G10"].iloc[0]]),
    pd.concat([data.xs("G21", level="sv").iloc[0], obs[obs["sv"]=="G21"].iloc[0]]),
    pd.concat([data.xs("G24", level="sv").iloc[0], obs[obs["sv"]=="G24"].iloc[0]]),
    pd.concat([data.xs("G17", level="sv").iloc[1], obs[obs["sv"]=="G17"].iloc[0]]),
    pd.concat([data.xs("G03", level="sv").iloc[1], obs[obs["sv"]=="G03"].iloc[0]]),
    pd.concat([data.xs("G14", level="sv").iloc[0], obs[obs["sv"]=="G14"].iloc[0]]),
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


for sat in satellites[5:6]:
    print(sat)
    t_s = T - sat["P"]/c + sat["dt"]

    t_k = t_s - sat["Toe"]
    if t_k >  302400: t_k -= 604800
    if t_k < -302400: t_k += 604800

    M_k = sat["M0"] + (GM**0.5 / sat["sqrtA"]**3 + sat["DeltaN"]) * t_k

    E_k = M_k
    for _ in range(3):
        E_k = E_k + (M_k - E_k + sat["Eccentricity"] * np.sin(E_k)) / (1 - sat["Eccentricity"] * np.cos(E_k))

    f_k = 2 * np.arctan(np.sqrt((1 + sat["Eccentricity"]) / (1 - sat["Eccentricity"])) * np.tan(E_k / 2))

    u_k = sat["omega"] + f_k + sat["Cuc"] * np.cos(2 * (sat["omega"] + f_k)) + sat["Cus"] * np.sin(2 * (sat["omega"] + f_k))

    r_k = sat["sqrtA"]**2 * (1 - sat["Eccentricity"] * np.cos(E_k)) + sat["Crc"] * np.cos(2 * (sat["omega"] + f_k)) + sat["Crs"] * np.sin(2 * (sat["omega"] + f_k))

    i_k = sat["Io"] + sat["IDOT"] * t_k + sat["Cic"] * np.cos(2 * (sat["omega"] + f_k)) + sat["Cis"] * np.sin(2 * (sat["omega"] + f_k))

    lambda_k = sat["Omega0"] + (sat["OmegaDot"] - omega_e) * t_k - omega_e * sat["Toe"]

    coords = R3(-lambda_k) @ R1(-i_k) @ R3(u_k) @ np.array([r_k, 0, 0])

    print("Coordinates (ECEF) at transmission time:")
    print(coords)
    print()
