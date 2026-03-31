# -*- coding: utf-8 -*-
"""
@author: heidrich
"""

import pandas as pd


class DualEdgeSplitter:
    """
    Dein aktuelles Setup:
    - Test vorne (erste Hälfte von test_fraction)
    - Test hinten (letzte Hälfte)
    - Train = Mitte
    """

    def __init__(self, date_col="date"):
        self.date_col = date_col

    def split(self, df: pd.DataFrame, test_fraction: float):
        df = df.sort_values(self.date_col).reset_index(drop=True)
        test_each = int(len(df) * test_fraction / 2)

        df_test_start = df.iloc[:test_each]
        df_test_end = df.iloc[-test_each:]
        df_train = df.iloc[test_each:-test_each]
        df_test = pd.concat([df_test_start, df_test_end]).reset_index(
            drop=True
        )

        return df_train, df_test, df_test_start, df_test_end
