# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:01:18 2022

@author: Antoine
"""
from pathlib import Path
import pandas as pd

CURRENT_DIR = Path(__file__).parent.absolute()

heart_data = pd.read_csv("heart.csv")