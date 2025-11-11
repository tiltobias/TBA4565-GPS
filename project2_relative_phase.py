import pandas as pd
import numpy as np
from project1_absolute_code import geodetic_to_cartesian

from data_project2_relative_phase.variables import T1, T2, base_position_llh, rover_approx_position_llh

base  = { "T1": pd.read_csv("data_project2_relative_phase/base_T1.csv"),
          "T2": pd.read_csv("data_project2_relative_phase/base_T2.csv")}
rover = { "T1": pd.read_csv("data_project2_relative_phase/rover_T1.csv"),
          "T2": pd.read_csv("data_project2_relative_phase/rover_T2.csv")}

"""
Step 1: Transform receiver to Cartesian coordinates
"""

base_position_xyz = geodetic_to_cartesian(base_position_llh)
rover_approx_position_xyz = geodetic_to_cartesian(rover_approx_position_llh)
print("Base position (XYZ):", base_position_xyz)
print("Rover approximate position (XYZ):", rover_approx_position_xyz)

