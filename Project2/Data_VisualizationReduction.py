# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:01:18 2022

@author: Antoine
"""
from pathlib import Path
import pandas as pd
# import numpy as np
CURRENT_DIR = Path(__file__).parent.absolute()

names_list = ["age", "sex", "chest pain type", "bp", "serum cholestoral", "blood sugar > 120 mg/dl", "resting ST", "max bpm", "ex-ind angina", "ex-ind ST depression", "ex ST slope", "fluoroscopy", "thalassemia", "heart disease Y/N" ]
heart_data = pd.read_csv("heart.csv", names = names_list)
heart_data.mean()