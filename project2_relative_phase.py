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
        if np.linalg.norm(delta_X[:3]) < 1e-6: break
        if i == 9: 
            print("Max iterations reached")
            break
    phase_ambiguities = delta_X[3:].flatten()
    C_X = np.linalg.inv(A.T @ P_matrix @ A)
    v = delta_L - A @ delta_X
    return rover_xyz, phase_ambiguities, C_X, v

def estimate_position_fixed(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz, fixed_ambiguities):
    rover_xyz_fixed = rover_approx_xyz.copy()
    for i in range(10):
        A = A_matrix(satellites_rover, rover_xyz_fixed, fixed_ambiguities=True)
        delta_L = delta_L_vector(satellites_base, satellites_rover, base_known_xyz, rover_xyz_fixed)
        for j in range(len(fixed_ambiguities)): # Subtract the fixed ambiguities contribution
            delta_L[j] -= fixed_ambiguities[j] * wavelength_L1
            delta_L[j + len(fixed_ambiguities)] -= fixed_ambiguities[j] * wavelength_L1
        delta_X = np.linalg.inv(A.T @ P_matrix @ A) @ A.T @ P_matrix @ delta_L
        rover_xyz_fixed += delta_X.flatten()
        if np.linalg.norm(delta_X) < 1e-6: break
        if i == 9: 
            print("Max iterations reached")
            break
    C_X = np.linalg.inv(A.T @ P_matrix @ A)
    v = delta_L - A @ delta_X
    return rover_xyz_fixed, C_X, v


def main():
    satellites_base  = {T1: pd.read_csv("data_project2_relative_phase/base_T1.csv"),
                        T2: pd.read_csv("data_project2_relative_phase/base_T2.csv")}
    satellites_rover = {T1: pd.read_csv("data_project2_relative_phase/rover_T1.csv"),
                        T2: pd.read_csv("data_project2_relative_phase/rover_T2.csv")}
    
    """
    Step 1: Transform receiver to Cartesian coordinates
    """
    print("Step 1:")

    base_known_xyz = geodetic_to_cartesian(base_known_llh)
    rover_approx_xyz = geodetic_to_cartesian(rover_approx_llh)
    print("Base position (XYZ):", base_known_xyz)
    print("Rover approximate position (XYZ):", rover_approx_xyz)


    """
    Step 2: Observation equation, design matrix and delta L vector. Estimate rover position with Double difference.
    """
    print("Step 2:")

    rover_xyz_float, phase_ambiguities, C_X_float, v = estimate_position_float(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz)

    print("Final rover coordinates in Cartesian:", rover_xyz_float)
    print("Estimated phase ambiguities (in cycles):", phase_ambiguities)
    print("Covariance matrix C_X:", C_X_float)
    print("Residuals vector v:", v.flatten())
    ssr = v.T @ P_matrix @ v
    print("Sum of squared residuals (SSR):", ssr)


    """
    Step 3: Fixing the ambiguities and re-estimating the rover position
    """
    print("Step 3a:")
    
    fixed_ambiguities_real = phase_ambiguities.copy() # Real ambiguities
    print("Fixed ambiguities (in cycles) (real)   :", fixed_ambiguities_real)
    rover_xyz_fixed_real, C_X, v = estimate_position_fixed(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz, fixed_ambiguities_real)
    print("Rover:", rover_xyz_fixed_real, "C_X:", C_X)


    print("Step 3b:")

    fixed_ambiguities_rounded = np.round(phase_ambiguities) # Round to nearest integer
    print("Fixed ambiguities (in cycles) (rounded):", fixed_ambiguities_rounded)
    rover_xyz_fixed_rounded, C_X, v = estimate_position_fixed(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz, fixed_ambiguities_rounded)
    print("Rover:", rover_xyz_fixed_rounded, "C_X:", C_X)

    print("Step 3c:")

    ambiguities_std = np.sqrt(np.diag(C_X_float)[3:])
    print("Ambiguity standard deviations (in cycles):", ambiguities_std)
    ambiguities_min = phase_ambiguities - 3 * ambiguities_std
    ambiguities_max = phase_ambiguities + 3 * ambiguities_std
    fixed_ambiguities_min = np.floor(ambiguities_min)
    fixed_ambiguities_max = np.ceil(ambiguities_max)

    def product_ranges(ranges):
        if not ranges:
            yield ()
            return
        first, *rest = ranges
        for value in first:
            for prod in product_ranges(rest):
                yield (value, ) + prod

    ranges = [
        range(int(min_i), int(max_i) + 1)
        for min_i, max_i in zip(fixed_ambiguities_min, fixed_ambiguities_max)
    ]
    print("Searching over ranges for fixed ambiguities:")
    for r in ranges:
        print(r, end=" ")
    print()

    ssr_best = float('inf')
    combination_best = None
    rover_xyz_best, C_X_best, v_best = None, None, None
    ssr_2best = float('inf')
    combination_2best = None
    rover_xyz_2best, C_X_2best, v_2best = None, None, None
    for fixed_ambiguities_combo in product_ranges(ranges):
        rover_xyz, C_X, v = estimate_position_fixed(satellites_base, satellites_rover, base_known_xyz, rover_approx_xyz, fixed_ambiguities_combo)
        ssr = (v.T @ P_matrix @ v).item()
        if ssr < ssr_best:
            ssr_2best = ssr_best
            combination_2best = combination_best
            rover_xyz_2best, C_X_2best, v_2best = rover_xyz_best, C_X_best, v_best
            ssr_best = ssr
            combination_best = fixed_ambiguities_combo
            rover_xyz_best, C_X_best, v_best = rover_xyz, C_X, v
        elif ssr < ssr_2best:
            ssr_2best = ssr
            combination_2best = fixed_ambiguities_combo
            rover_xyz_2best, C_X_2best, v_2best = rover_xyz, C_X, v


    """
    Step 4: Convert final rover position to Geodetic coordinates
    """
    print("Step 4:")

    rover_llh_float = cartesian_to_geodetic(rover_xyz_float)
    print("Rover coordinates float solution            (lat, lon, height):", rover_llh_float)

    rover_llh_fixed_real = cartesian_to_geodetic(rover_xyz_fixed_real)
    print("Rover coordinates real fixed ambiguities    (lat, lon, height):", rover_llh_fixed_real)

    rover_llh_fixed_rounded = cartesian_to_geodetic(rover_xyz_fixed_rounded)
    print("Rover coordinates rounded fixed ambiguities (lat, lon, height):", rover_llh_fixed_rounded)

    rover_llh_best = cartesian_to_geodetic(rover_xyz_best)
    print("Best rover coordinates in Geodetic          (lat, lon, height):", rover_llh_best)
    print("Best fixed ambiguities (in cycles)          :", combination_best)

    rover_llh_2best = cartesian_to_geodetic(rover_xyz_2best)
    print("2nd Best rover coordinates in Geodetic      (lat, lon, height):", rover_llh_2best)
    print("2nd Best fixed ambiguities (in cycles)      :", combination_2best)

    print("ssr ratio 2nd best / best:", ssr_2best / ssr_best)
    print("Best solution likely correct if ratio > 3:", ssr_2best / ssr_best > 3)


if __name__ == "__main__":
    main()