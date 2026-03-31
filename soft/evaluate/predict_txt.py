# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:00:02 2026

@author: heidrich
"""

import lightgbm as lgb
import pandas as pd

# Modell laden
booster = lgb.Booster(model_file="Aalborg_SFH_hourly__split01.txt")

# Wichtig: exakt dieselben Feature-Spalten wie im Modell (Reihenfolge egal bei DataFrame,
# aber Namen müssen passen)
feature_names = booster.feature_name()

# X muss diese Spalten enthalten
X = df[feature_names]  # df = dein DataFrame mit Features

y_pred = booster.predict(X)
