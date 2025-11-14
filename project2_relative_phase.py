import pandas as pd
import numpy as np
from project1_absolute_code import geodetic_to_cartesian, cartesian_to_geodetic

from data_project2_relative_phase.variables import T1, T2, base_known_llh, rover_approx_llh, P_matrix

"""
General constants
"""
c = 299792458
frequency_L1 = 1575.42e6
wavelength_L1 = c / frequency_L1


"""
Functions
"""

def A_matrix(satellites_rover, approx_rover_xyz, fixed_ambiguities=False):
    A = []
    for T in (T1, T2):
        sat_rover_i = satellites_rover[T].iloc[0]
        rho_B_i = np.linalg.norm(sat_rover_i[["X", "Y", "Z"]].values - approx_rover_xyz[:3])
        for j in range(1, len(satellites_rover[T])):
            sat_rover = satellites_rover[T].iloc[j]
            rho_B_j = np.linalg.norm(sat_rover[["X", "Y", "Z"]].values - approx_rover_xyz[:3])
            
            if not fixed_ambiguities:
                row = np.zeros(3 + len(satellites_rover[T]) -1)
                row[3 + j -1] = wavelength_L1
            else:
                row = np.zeros(3)
            
            for x, X in enumerate(["X","Y","Z"]):
                row[x] = ( -( (sat_rover[X] - approx_rover_xyz[x]) / rho_B_j ) + ( (sat_rover_i[X] - approx_rover_xyz[x]) / rho_B_i ) )
            A.append(row)
    return np.array(A)

def delta_L_vector(satellites_base, satellites_rover, approx_base_xyz, approx_rover_xyz):
    L = []
    for T in (T1, T2):
        sat_base_i = satellites_base[T].iloc[0]
        sat_rover_i = satellites_rover[T].iloc[0]
        rho_A_i = np.linalg.norm(sat_base_i[["X", "Y", "Z"]].values - approx_base_xyz)
        rho_B_i = np.linalg.norm(sat_rover_i[["X", "Y", "Z"]].values - approx_rover_xyz)
        for j in range(1, len(satellites_rover[T])):
            sat_base = satellites_base[T].iloc[j]
            sat_rover = satellites_rover[T].iloc[j]
            rho_A_j = np.linalg.norm(sat_base[["X", "Y", "Z"]].values - approx_base_xyz)
            rho_B_j = np.linalg.norm(sat_rover[["X", "Y", "Z"]].values - approx_rover_xyz)

            Phi_AB_ij = (sat_rover["L1"] - sat_rover_i["L1"] - sat_base["L1"] + sat_base_i["L1"]) * wavelength_L1
            L.append(Phi_AB_ij - rho_B_j + rho_B_i + rho_A_j - rho_A_i)
    return np.array(L).reshape(-1, 1)

def estimate_position_float(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz):
    rover_xyz = rover_approx_xyz.copy()
    for i in range(10):
        A = A_matrix(satellites_rover, rover_xyz)
        delta_L = delta_L_vector(satellites_base, satellites_rover, base_known_xyz, rover_xyz)
        delta_X = np.linalg.inv(A.T @ P_matrix @ A) @ A.T @ P_matrix @ delta_L
        rover_xyz += delta_X[:3].flatten()
        phase_ambiguities = delta_X[3:].flatten()
        C_X = np.linalg.inv(A.T @ P_matrix @ A)
        if np.linalg.norm(delta_X[:3]) < 1e-6: break
        if i == 9: 
            print("Max iterations reached")
            break
    return rover_xyz, phase_ambiguities, C_X

def estimate_position_fixed(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz, fixed_ambiguities):
    rover_xyz_fixed = rover_approx_xyz.copy()
    for i in range(10):
        A = A_matrix(satellites_rover, rover_xyz_fixed, fixed_ambiguities=True)
        delta_L = delta_L_vector(satellites_base, satellites_rover, base_known_xyz, rover_xyz_fixed)
        for j in range(len(fixed_ambiguities)): # Subtract the fixed ambiguities contribution
            delta_L -= fixed_ambiguities[j] * wavelength_L1
        delta_X = np.linalg.inv(A.T @ P_matrix @ A) @ A.T @ P_matrix @ delta_L
        rover_xyz_fixed += delta_X.flatten()
        C_X = np.linalg.inv(A.T @ P_matrix @ A)
        if np.linalg.norm(delta_X) < 1e-6: break
        if i == 9: 
            print("Max iterations reached")
            break
    return rover_xyz_fixed, C_X


def main():
    satellites_base  = {T1: pd.read_csv("data_project2_relative_phase/base_T1.csv"),
                        T2: pd.read_csv("data_project2_relative_phase/base_T2.csv")}
    satellites_rover = {T1: pd.read_csv("data_project2_relative_phase/rover_T1.csv"),
                        T2: pd.read_csv("data_project2_relative_phase/rover_T2.csv")}
    
    """
    Step 1: Transform receiver to Cartesian coordinates
    """

    base_known_xyz = geodetic_to_cartesian(base_known_llh)
    rover_approx_xyz = geodetic_to_cartesian(rover_approx_llh)
    print("Base position (XYZ):", base_known_xyz)
    print("Rover approximate position (XYZ):", rover_approx_xyz)
    # TODO: rename to llh and xyz in project1


    """
    Step 2: Observation equation, design matrix and delta L vector. Estimate rover position with Double difference.
    """
    print("Step 2:")

    rover_xyz, phase_ambiguities, C_X = estimate_position_float(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz)

    print("Final rover coordinates in Cartesian:", rover_xyz)
    print("Estimated phase ambiguities (in cycles):", phase_ambiguities)
    print("Covariance matrix C_X:", C_X[:3, :3].diagonal())


    """
    Step 3: Fixing the ambiguities and re-estimating the rover position
    """
    print("Step 3:")
    
    fixed_ambiguities = phase_ambiguities.copy() # Real ambiguities
    print("Fixed ambiguities (in cycles):", fixed_ambiguities)
    rover_xyz_fixed, C_X = estimate_position_fixed(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz, fixed_ambiguities)
    print("Final rover coordinates with fixed ambiguities in Cartesian:", rover_xyz_fixed)
    print("Covariance matrix C_X:", C_X.diagonal())

    fixed_ambiguities = np.round(phase_ambiguities) # Round to nearest integer
    print("Fixed ambiguities (in cycles):", fixed_ambiguities)
    rover_xyz_fixed, C_X = estimate_position_fixed(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz, fixed_ambiguities)
    print("Final rover coordinates with fixed ambiguities in Cartesian:", rover_xyz_fixed)
    print("Covariance matrix C_X:", C_X.diagonal())


    """
    Step 4: Convert final rover position to Geodetic coordinates
    """
    print("Step 4:")

    rover_llh = cartesian_to_geodetic(rover_xyz)
    print("Final rover coordinates in Geodetic (lat, lon, height):", rover_llh)

    rover_llh_fixed = cartesian_to_geodetic(rover_xyz_fixed)
    print("Final rover coordinates in Geodetic (lat, lon, height):", rover_llh_fixed)


if __name__ == "__main__":
    main()