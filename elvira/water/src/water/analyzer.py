import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from  pathlib import Path
from typing import Any, Dict, Optional

class WeatherDataAnalyzer:
    """
        Class for advanced agro-meteorological analysis of data series.

        -   Loading data
        -   Development of daytime profiles
        -   Visualization of water stress and evapotranspiration metrics
    """

    REQUIRED_COLUMNS = ["ts", "water_potential_last", "temp_out", "et", "solar_rad_avg"]

    def __init__(self, file_path: str | Path):
        """
        Analyzer constructur with the path of .csv file

        Args:
        -   file_path: string of data series
        """

        self.file_path = Path(file_path)
        self.df = Optional[pd.DataFrame] = None
        self._validate_file()

    def _validate_file(self) -> None:
        """
        File validation method.
        Check if the file exists and its extansion.
        """

        if not self.file_path:
            raise FileNotFoundError(
                f'Error: File {self.file_path} not found or not exists.'
            )
        if self.file_path.suffix.lower() != ".csv":
            raise ValueError(
                'Error: file is not a CSV.'
            )
            
    def load_data(self) -> WeatherDataAnalyzer:
        """
        Load data from .csv

        -   Columns validation
        -   Indexing data series
        """

        self.df = pd.read_csv(self.file_path)

        missing_cols = [
            col for col in self.REQUIRED_COLUMNS if col not in self.df.columns
        ]

        if missing_cols:
            raise KeyError(
                f'Missing columns : {missing_cols}. Impossible to procede.'
            )
        
        self.df['ts'] = pd.to_datetime(self.df['ts'])
        self.df['hour'] = self.df['ts'].dt.hour
        self.dt['date'] = self.df['ts'].dt.date

        return self

    def compute_hourly_profile(self) -> pd.DataFrame:
        """
        Compute the hourly profile (24 hours pattern) with histrical data aggragation.

        Return:
        -   pd.DataFrame: aggregate DataFrame by hour (0-23) with the average of feature.
        """         

        if self.df is None:
            raise ValueError(
                "Data not loaded. Calle load_data() before."
            )
        
        hourly_profile = (
            self.df.groupby('hour')
            .agg(
                #water_potential=("water_potential_lase", "mean"),
                temp_out=('temp_out', 'mean'),
                et=('et', 'mean'),
                solar_rad=('solar_rad_avg', 'mean')
            )
            .reset_index()            
        )

        return hourly_profile