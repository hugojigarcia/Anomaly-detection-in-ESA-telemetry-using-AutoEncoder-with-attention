import pandas as pd

def calculate_adtqc_metrics(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame):
    return {
        "adtqc_after_ratio": 0,
        "adtqc_score": 0
    }




#%%
# import abc
# import warnings
# import math
# from typing import Optional, Tuple, Dict

# import numpy as np
# import pandas as pd
# import portion as P
# from sklearn.utils import assert_all_finite, check_consistent_length

# # from .utils import convert_time_series_to_events



#%%
# def convert_time_series_to_events(vector=[[pd.to_datetime("2015-01-01"), 0], [pd.to_datetime("2015-01-05"), 1], [pd.to_datetime("2015-01-10"), 1],
#                          [pd.to_datetime("2015-01-18"), 1], [pd.to_datetime("2015-01-20"), 0]]):
#     """
#     Convert time series (a list of timestamps and values > 0 indicating for the anomalous instances)
#     to a list of events. The events are considered as durations,
#     i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).

#     :param vector: a list of elements belonging to {0, 1}
#     :return: a list of couples, each couple representing the start and stop of
#     each event
#     """
#     vector = np.asarray(vector)

#     def find_runs(x):
#         """Find runs of consecutive items in an array."""

#         # ensure array
#         x = np.asanyarray(x)
#         if x.ndim != 1:
#             raise ValueError('only 1D array supported')
#         n = x.shape[0]

#         # handle empty array
#         if n == 0:
#             return np.array([]), np.array([]), np.array([])

#         else:
#             # find run starts
#             loc_run_start = np.empty(n, dtype=bool)
#             loc_run_start[0] = True
#             np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
#             run_starts = np.nonzero(loc_run_start)[0]

#             # find run values
#             run_values = x[loc_run_start]

#             # find run lengths
#             run_lengths = np.diff(np.append(run_starts, n))

#             run_ends = run_starts + run_lengths

#             return np.stack((run_starts[run_values > 0], run_ends[run_values > 0])).transpose()

#     non_zero_runs = find_runs(vector[..., 1])

#     events = []
#     n = len(vector)
#     for x, y in non_zero_runs:
#         if y == n:
#             events.append(P.closed(vector[..., 0][x], vector[..., 0][y - 1]))
#         else:
#             events.append(P.closedopen(vector[..., 0][x], vector[..., 0][y]))
#     events = P.Interval(*events)

#     return events



# %%
# class MultiChannelMetric(abc.ABC):
#     """Base class for metric implementations that assess a quality of anomalous channels identification.

#     Examples
#     --------
#     You can implement a new TimeEval metric easily by inheriting from this base class. A simple metric, for example,
#     uses a fixed threshold to get binary labels and computes the false positive rate:

#     """

#     def __call__(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:  # type: ignore[no-untyped-def]
#         y_true, y_score = self._validate_scores(y_true, y_score, **kwargs)
#         if np.unique(y_score).shape[0] == 1:
#             warnings.warn("Cannot compute metric for a constant value in y_score, returning 0.0!")
#             return 0.
#         return self.score(y_true, y_score)

#     def _validate_scores(self, y_true: np.ndarray, y_score: np.ndarray,
#                          inf_is_1: bool = True,
#                          neginf_is_0: bool = True,
#                          nan_is_0: bool = True) -> Tuple[np.ndarray, np.ndarray]:
#         # check labels
#         if y_true.dtype.kind == "f" and y_score.dtype.kind in ["i", "u"]:
#             warnings.warn("Assuming that y_true and y_score where permuted, because their dtypes indicate so. "
#                           "y_true should be an integer array and y_score a float array!")
#             return self._validate_scores(y_score, y_true)

#         assert_all_finite(y_true)

#         check_consistent_length([y_true, y_score])

#         # substitute NaNs and Infs
#         nan_mask = np.isnan(y_score)
#         inf_mask = np.isinf(y_score)
#         neginf_mask = np.isneginf(y_score)
#         penalize_mask = np.full_like(y_score, dtype=bool, fill_value=False)
#         if inf_is_1:
#             y_score[inf_mask] = 1
#         else:
#             penalize_mask = penalize_mask | inf_mask
#         if neginf_is_0:
#             y_score[neginf_mask] = 0
#         else:
#             penalize_mask = penalize_mask | neginf_mask
#         if nan_is_0:
#             y_score[nan_mask] = 0.
#         else:
#             penalize_mask = penalize_mask | nan_mask
#         y_score[penalize_mask] = (~np.array(y_true[penalize_mask], dtype=bool)).astype(np.int_)

#         assert_all_finite(y_score)
#         return y_true, y_score

#     @property
#     @abc.abstractmethod
#     def name(self) -> str:
#         """Returns the unique name of this metric."""
#         ...

#     @abc.abstractmethod
#     def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
#         """Implementation of the metric's scoring function."""
#         ...

#     @abc.abstractmethod
#     def supports_continuous_scorings(self) -> bool:
#         """Whether this metric accepts continuous anomaly scorings as input (``True``) or binary classification
#         labels (``False``)."""
#         ...



# %%
# class ADTQC(MultiChannelMetric):
#     """Computes anomaly detection timing quality curve (ADTQC) scores used in the ESA Anomaly Detection Benchmark.

#     Parameters
#     ----------
#     exponent : float
#         Value of exponent of ADTQC. The default is math.e
#     full_range : tuple of datetimes
#         Optional tuple of (start time, end time) of the original data.
#         If None, it is automatically inferred from the data.
#     select_labels : dict
#         Optional dictionary of event categories, classes or types to include in the calculation.
#         Dictionary should contain column names and values from anomaly_types.csv as keys and values.
#         If None, all events are included.
#     name : str
#         Optional custom name for the metric.
#     """

#     def __init__(self,
#                  exponent: float = math.e,
#                  full_range: Optional[tuple] = None,
#                  select_labels: Optional[dict] = None,
#                  name: Optional[str] = None) -> None:
#         self.exponent = exponent
#         self.full_range = full_range

#         if select_labels is None or len(select_labels) == 0:
#             self.selected_labels = dict()
#             filter_string = "ALL"
#         else:
#             select_labels = {col: np.atleast_1d(val) for col, val in select_labels.items()}
#             self.selected_labels = select_labels
#             filter_string = "_".join(["_".join(val) for val in select_labels.values()])
#         self._name = f"ADTQC_{filter_string}" if name is None else name

#     def timing_curve(self, x, a, b):
#         assert a >= pd.Timedelta(0)
#         assert b >= pd.Timedelta(0)
#         if (a == pd.Timedelta(0) or b == pd.Timedelta(0)) and x == pd.Timedelta(0):
#             return 1
#         if x <= -a or x >= b:
#             return 0
#         if -a < x <= pd.Timedelta(0):
#             return ((x + a)/a)**self.exponent
#         if pd.Timedelta(0) < x < b:
#             denom_part = x/(b - x)
#             return 1. / (1. + denom_part**self.exponent)

#     def score(self, y_true: pd.DataFrame, y_pred: dict) -> dict:
#         """
#         Calculate scores.
#         :param y_true: DataFrame representing labels.csv from ESA-ADB
#         :param y_pred: dict of {channel_name: list of pairs (timestamp, is_anomaly)}, where is_anomaly is binary, 0 - nominal, 1 - anomaly
#         :return: dictionary of calculated scores
#         """
#         for channel, values in y_pred.items():
#             y_pred[channel] = np.asarray(values)

#         # Adjust to full range
#         min_y_pred = min(np.concatenate([y[..., 0] for y in y_pred.values()]))
#         max_y_pred = max(np.concatenate([y[..., 0] for y in y_pred.values()]))
#         if self.full_range is None:  # automatic full range
#             self.full_range = (min(y_true["StartTime"].min(), min_y_pred), max(y_true["EndTime"].max(), max_y_pred))
#         else:
#             assert self.full_range[0] <= y_true["StartTime"].min()
#             assert self.full_range[1] >= y_true["EndTime"].max()
#             assert self.full_range[0] <= min_y_pred
#             assert self.full_range[1] >= max_y_pred

#         for channel, values in y_pred.items():
#             if y_pred[channel][0, 0] > self.full_range[0]:
#                 y_pred[channel] = np.array([np.array([self.full_range[0], y_pred[channel][0, 1]]), *y_pred[channel]])
#             if y_pred[channel][-1, 0] < self.full_range[1]:
#                 y_pred[channel] = np.array([*y_pred[channel], np.array([self.full_range[1], y_pred[channel][-1, 1]])])

#         # Find prediction intervals per channel
#         events_pred_dict = dict()
#         for channel, pred in y_pred.items():
#             events_pred_dict[channel] = convert_time_series_to_events(np.asarray(pred))

#         # Analyze only selected anomaly types
#         filtered_y_true = y_true.copy()
#         for col, val in self.selected_labels.items():
#             filtered_y_true = filtered_y_true[filtered_y_true[col].isin(val)]

#         unique_anomaly_ids = filtered_y_true["ID"].unique()
#         start_times = []
#         for aid in unique_anomaly_ids:
#             gt = filtered_y_true[filtered_y_true["ID"] == aid]
#             start_times.append(min(gt["StartTime"]))
#         start_times = sorted(start_times)

#         before_tps = []
#         after_tps = []
#         curve_scores = []
#         for aid in unique_anomaly_ids:
#             gt = filtered_y_true[filtered_y_true["ID"] == aid]

#             affected_channels = np.sort(gt["Channel"].unique())
#             channels_intervals = dict()
#             for channel in affected_channels:
#                 c_gt = gt[gt["Channel"] == channel]
#                 c_gt_intervals = []
#                 for _, row in c_gt[["StartTime", "EndTime"]].iterrows():
#                     c_gt_intervals.append(P.closed(*row))
#                 channels_intervals[channel] = P.Interval(*c_gt_intervals)

#             global_preds = []
#             global_gts = []
#             for channel in affected_channels:
#                 if channel in events_pred_dict.keys():
#                     events_pred = [pred for pred in events_pred_dict[channel] if not (pred & channels_intervals[channel]).empty]
#                     # print(f"{events_pred_dict=}")
#                     # events_pred = [pred for pred in events_pred_dict[channel]]
#                     global_preds.extend(events_pred)
#                     global_gts.append(channels_intervals[channel])
#             global_preds = P.Interval(*global_preds)
#             if global_preds.empty:  # no detection no score
#                 continue
#             global_gts = P.Interval(*global_gts)

#             anomaly_length = global_gts.upper - global_gts.lower
#             print(f"{global_gts.lower=}")
#             current_anomaly_idx = start_times.index(global_gts.lower)
#             previous_anomaly_start = start_times[current_anomaly_idx - 1] if current_anomaly_idx > 0 else global_gts.lower - anomaly_length
#             alpha = min(anomaly_length, global_gts.lower - previous_anomaly_start)

#             latency = global_preds.lower - global_gts.lower
#             metric_value = self.timing_curve(latency, alpha, anomaly_length)
#             curve_scores.append(metric_value)

#             if latency < pd.Timedelta(0):
#                 before_tps.append(metric_value)
#             else:
#                 after_tps.append(metric_value)

#         print(before_tps)
#         print(after_tps)
#         print(curve_scores)

#         before_tps = np.array(before_tps)
#         after_tps = np.array(after_tps)
#         curve_scores = np.array(curve_scores)

#         result_dict = {"Nb_Before": len(before_tps),
#                        "Nb_After": len(after_tps),
#                        "AfterRate": len(after_tps) / len(curve_scores) if len(curve_scores) > 0 else np.nan,
#                        "Total": np.mean(curve_scores) if len(curve_scores) > 0 else np.nan}

#         return result_dict

#     def supports_continuous_scorings(self) -> bool:
#         return False

#     @property
#     def name(self) -> str:
#         return self._name



# %%
# def calculate_adtqc_scores(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame) -> dict:
#     # 1. Procesar grouped_anomalies para convertirlo en el formato necesario para ADTQC
#     # Asumimos que cada fila es una predicción de anomalía para un canal específico
    
#     # Convertir a formato esperado por ADTQC (dict con listas de pares de timestamp, is_anomaly)
#     y_pred = {}
#     for _, row in grouped_anomalies.iterrows():
#         channel = process_pred_channels(row["Ranking_cols"])[0]  # Tomamos el primer canal en la lista de Ranking_cols
#         if channel not in y_pred:
#             y_pred[channel] = []
#         y_pred[channel].append([pd.to_datetime(row["time_start"]), int(row["is_anomaly"])])

#     # Convertimos los valores a arrays
#     for channel in y_pred:
#         y_pred[channel] = np.array(y_pred[channel])

#     # 2. Procesar true_anomalies para adaptarlo al formato de y_true que espera ADTQC
#     y_true = true_anomalies[["ID", "Channel", "StartTime", "EndTime"]].copy()
    
#     # Asegurar que las columnas de tiempo sean de tipo datetime
#     y_true["StartTime"] = pd.to_datetime(y_true["StartTime"])
#     y_true["EndTime"] = pd.to_datetime(y_true["EndTime"])

#     # 3. Crear una instancia de ADTQC con los parámetros necesarios
#     # Usamos el rango temporal total (full_range) de los datos en true_anomalies
#     full_range = (y_true["StartTime"].min(), y_true["EndTime"].max())
#     metric = ADTQC(full_range=full_range)

#     # 4. Calcular el score
#     result = metric.score(y_true, y_pred)

#     # Devolvemos el resultado con el AfterRate y el Total score
#     return {
#         "ADTQC After Ratio": result["AfterRate"],
#         "ADTQC Score": result["Total"]
#     }
# calculate_adtqc_scores(grouped_anomalies, true_anomalies)