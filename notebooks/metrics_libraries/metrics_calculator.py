import pandas as pd
import numpy as np

from esa_libraries.ESAScores import ESAScores
from esa_libraries.ADTQC import ADTQC
from esa_libraries.ChannelAwareFScore import ChannelAwareFScore


class MetricsCalculator():
    def __init__(self, esa_anomalies_path, channels_info_path, channels_list=None):
        # Load ESA anomalies
        self.esa_anomalies_path = esa_anomalies_path
        self.esa_anomalies = pd.read_csv(self.esa_anomalies_path)
        self.esa_anomalies['StartTime'] = pd.to_datetime(self.esa_anomalies['StartTime'], format='mixed', errors='coerce')
        self.esa_anomalies['StartTime'] = self.esa_anomalies['StartTime'].dt.tz_localize(None)
        self.esa_anomalies['EndTime'] = pd.to_datetime(self.esa_anomalies['EndTime'], format='mixed', errors='coerce')
        self.esa_anomalies['EndTime'] = self.esa_anomalies['EndTime'].dt.tz_localize(None)
        if channels_list is not None:
            self.esa_anomalies = self.esa_anomalies[self.esa_anomalies["Channel"].isin(channels_list)]

        # Load channels info
        self.channels_info_path = channels_info_path
        channels_info = pd.read_csv(self.channels_info_path)
        self.subsystems_mapping = channels_info.groupby("Subsystem")["Channel"].apply(list).to_dict()
    
    def __get_metrics_calculators(self, full_range):
        # scores_calculator = ESAScores(betas=0.5, full_range=full_range, select_labels={"Dimensionality": "Multivariate", "Length": "Subsequence"})
        scores_calculator = ESAScores(betas=0.5, full_range=full_range)
        adtqc_calculator = ADTQC(full_range=full_range)
        aware_calculator = ChannelAwareFScore(full_range=full_range)
        return scores_calculator, adtqc_calculator, aware_calculator
    
    # def import_esa_anomalies_for_metrics(self, esa_anomalies_path, start_date, end_date):
    #     # Load ESA anomalies
    #     esa_anomalies = pd.read_csv(esa_anomalies_path)
    #     esa_anomalies['StartTime'] = pd.to_datetime(esa_anomalies['StartTime'], format='mixed', errors='coerce')
    #     esa_anomalies['StartTime'] = esa_anomalies['StartTime'].dt.tz_localize(None)
    #     esa_anomalies['EndTime'] = pd.to_datetime(esa_anomalies['EndTime'], format='mixed', errors='coerce')
    #     esa_anomalies['EndTime'] = esa_anomalies['EndTime'].dt.tz_localize(None)

    #     # Filter by date
    #     esa_anomalies = esa_anomalies[(esa_anomalies["EndTime"] >= start_date) & (esa_anomalies["StartTime"] <= end_date)]
    #     esa_anomalies.reset_index(drop=True, inplace=True)
    #     return esa_anomalies
    
    def filter_esa_anomalies(self, start_date, end_date):
        # Filter by date
        esa_anomalies_filtered = self.esa_anomalies[(self.esa_anomalies["EndTime"] >= start_date) & (self.esa_anomalies["StartTime"] <= end_date)]
        esa_anomalies_filtered.loc[esa_anomalies_filtered['StartTime'] < start_date, 'StartTime'] = start_date
        esa_anomalies_filtered.loc[esa_anomalies_filtered['EndTime'] > end_date, 'EndTime'] = end_date
        esa_anomalies_filtered.reset_index(drop=True, inplace=True)
        return esa_anomalies_filtered
    
    def import_anomalies_df(self, anomalies_path, start_date, end_date):
        anomalies_df = pd.read_csv(anomalies_path, index_col=0)
        anomalies_df.index = pd.to_datetime(anomalies_df.index)
        return anomalies_df[(anomalies_df.index >= start_date) & (anomalies_df.index <= end_date)]
    
    def get_anomalies_list(self, anomalies_df):
        # Genera una lista donde sus elementos son listas de dos elementos con el timestampt y 0 si todos los valores de la fila es 0 y 1 si alguno es 1
        anomalies_list = []
        for index, row in anomalies_df.iterrows():
            if row.any():
                anomalies_list.append([index, 1])
            else:
                anomalies_list.append([index, 0])
        return anomalies_list
    
    def get_anomalies_dict(self, anomalies_df):
        anomalies_dict = {}
        for col in anomalies_df.columns:
            values = np.array([[idx, val] for idx, val in anomalies_df[col].items()], dtype=object)
            anomalies_dict[col] = values
        return anomalies_dict



    def get_metrics(self, anomalies_df, start_date, end_date):
        # CREATE CLASSES
        scores_calculator, adtqc_calculator, aware_calculator = self.__get_metrics_calculators(full_range=(start_date, end_date))

        # IMPORT DATA
        esa_anomalies_filtered = self.filter_esa_anomalies(start_date, end_date)
        # anomalies_df = self.import_anomalies_df(anomalies_path, start_date, end_date)

        # FORMAT DATA
        anomalies_list= self.get_anomalies_list(anomalies_df)
        anomalies_dict = self.get_anomalies_dict(anomalies_df)

        # SCORES
        scores_metrics = scores_calculator.score(esa_anomalies_filtered, anomalies_list)
        adtqc_metrics = adtqc_calculator.score(esa_anomalies_filtered, anomalies_dict)
        aware_metrics = aware_calculator.score(esa_anomalies_filtered, anomalies_dict, self.subsystems_mapping)


        return scores_metrics | adtqc_metrics | aware_metrics
        # return scores_metrics | scores_metrics | scores_metrics
    
    def print_metrics_table(self, metrics: dict):
        print(f"|-----------------------------------------|")
        print(f"|     Event-wise    |  Precision  | {metrics['EW_precision']:.3f} |")
        print(f"|     Event-wise    |   Recall    | {metrics['EW_recall']:.3f} |")
        print(f"|     Event-wise    |    F0.5     | {metrics['EW_F_0.50']:.3f} |")
        print(f"|-----------------------------------------|")
        print(f"|  Subsystem-aware  |  Precision  | {metrics['subsystem_precision']:.3f} |")
        print(f"|  Subsystem-aware  |   Recall    | {metrics['subsystem_recall']:.3f} |")
        print(f"|  Subsystem-aware  |    F0.5     | {metrics['subsystem_F0.50']:.3f} |")
        print(f"|-----------------------------------------|")
        print(f"|   Channel-aware   |  Precision  | {metrics['channel_precision']:.3f} |")
        print(f"|   Channel-aware   |   Recall    | {metrics['channel_recall']:.3f} |")
        print(f"|   Channel-aware   |    F0.5     | {metrics['channel_F0.50']:.3f} |")
        print(f"|-----------------------------------------|")
        print(f"|       Alarming precision        | {metrics['alarming_precision']:.3f} |")
        print(f"|-----------------------------------------|")
        print(f"|       ADTQC       | After ratio | {metrics['AfterRate']:.3f} |")
        print(f"|       ADTQC       |    Score    | {metrics['Total']:.3f} |")
        print(f"|-----------------------------------------|")
        print(f"| Affiliation-based |  Precision  | {metrics['AFF_precision']:.3f} |")
        print(f"| Affiliation-based |   Recall    | {metrics['AFF_recall']:.3f} |")
        print(f"| Affiliation-based |    F0.5     | {metrics['AFF_F_0.50']:.3f} |")
        print(f"|-----------------------------------------|")