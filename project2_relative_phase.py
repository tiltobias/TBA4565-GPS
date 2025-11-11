import pandas as pd
import numpy as np
from project1_absolute_code import geodetic_to_cartesian

from data_project2_relative_phase.variables import T1, T2, base_known_llh, rover_approx_llh



# def rho(receiver, satellite, T):
#     rho_rec_sat = np.linalg.norm(satellite[["X","Y","Z"]].values - receiver)
#     return 

def A_matrix(satellites_rover, approx_rover_xyz):
    A = []
    for T in (T1, T2):
        sat_rover_i = satellites_rover[T].iloc[0]
        rho_B_i = np.linalg.norm([sat_rover_i["X"],sat_rover_i["Y"],sat_rover_i["Z"]] - approx_rover_xyz[:3])
        for n in range(1, len(satellites_rover[T])):
            sat_rover = satellites_rover[T].iloc[n]

            rho_B_j = np.linalg.norm([sat_rover["X"],sat_rover["Y"],sat_rover["Z"]] - approx_rover_xyz[:3])
            row = np.zeros(3 + len(satellites_rover[T]) -1)
            for x, X in enumerate(["X","Y","Z"]):
                row[x] = ( -( (sat_rover[X] - approx_rover_xyz[x]) / rho_B_j ) + ( (sat_rover_i[X] - approx_rover_xyz[x]) / rho_B_i ) )
            row[3 + n-1] = 1
            A.append(row)
    return np.array(A)



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
    print(A_matrix(satellites_rover, rover_approx_xyz))





if __name__ == "__main__":
    main()