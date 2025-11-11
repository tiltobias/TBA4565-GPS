import pandas as pd
import numpy as np

base  = { "T1": pd.read_csv("data_project2_relative_phase/base_T1.csv"),
          "T2": pd.read_csv("data_project2_relative_phase/base_T2.csv")}
rover = { "T1": pd.read_csv("data_project2_relative_phase/rover_T1.csv"),
          "T2": pd.read_csv("data_project2_relative_phase/rover_T2.csv")}


