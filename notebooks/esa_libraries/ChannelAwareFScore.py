from typing import Optional, Dict

import numpy as np
import pandas as pd
import portion as P

from esa_libraries.utils import convert_time_series_to_events


class ChannelAwareFScore():
    """Computes channel-aware and subsystem-aware F-scores used in the ESA Anomaly Detection Benchmark.

    Parameters
    ----------
    beta : float
        beta determines the weight of recall in the combined F-beta-scores.
        beta < 1 lends more weight to precision, while beta > 1 favors recall.
    select_labels : dict
        Optional dictionary of event categories, classes or types to include in the calculation.
        Dictionary should contain column names and values from anomaly_types.csv as keys and values.
        If None, all events are included.
    full_range : tuple of datetimes
        Optional tuple of (start time, end time) of the original data.
        If None, it is automatically inferred from the data.
    name : str
        Optional custom name for the metric.
    """
    def __init__(self,
                 beta: float = 0.5,
                 select_labels: Optional[dict] = None,
                 full_range: Optional[tuple] = None,
                 name: Optional[str] = None) -> None:
        self._beta = beta
        self.full_range = full_range

        if select_labels is None or len(select_labels) == 0:
            self.selected_labels = dict()
            filter_string = "ALL"
        else:
            select_labels = {col: np.atleast_1d(val) for col, val in select_labels.items()}
            self.selected_labels = select_labels
            filter_string = "_".join(["_".join(val) for val in select_labels.values()])
        self._name = f"PC_{filter_string}" if name is None else name

    def get_pr_re_f_score(self, true_positives, false_positives, false_negatives):
        divider = true_positives + false_positives
        if divider == 0:
            precision = 0.0
        else:
            precision = true_positives / divider

        divider = true_positives + false_negatives
        if divider == 0:
            recall = 0.0
        else:
            recall = true_positives / divider

        divider = (self._beta ** 2) * precision + recall
        if divider == 0:
            f_score = 0.0
        else:
            f_score = ((1 + self._beta ** 2) * precision * recall) / divider

        return precision, recall, f_score

    def score(self, y_true: pd.DataFrame, y_pred: dict, subsystems_mapping: dict = None) -> Dict[str, float]:
        """
        Calculate scores.
        :param y_true: DataFrame representing labels.csv from ESA-ADB
        :param y_pred: dict of {channel_name: list of pairs (timestamp, is_anomaly)}, where is_anomaly is binary, 0 - nominal, 1 - anomaly
        :param subsystems_mapping: dict of {subsystem_name: list of channel_names}
        :return: dictionary of calculated scores
        """
        all_channels = list(y_pred.keys())

        for c, preds in y_pred.items():
            y_pred[c] = np.asarray(preds)

        min_y_pred_timestamp = min([preds[..., 0].min() for preds in y_pred.values()])
        max_y_pred_timestamp = max([preds[..., 0].max() for preds in y_pred.values()])
        if self.full_range is None:  # automatic full range
            self.full_range = (min(y_true["StartTime"].min(), min_y_pred_timestamp), max(y_true["EndTime"].max(), max_y_pred_timestamp))
        else:
            assert self.full_range[0] <= y_true["StartTime"].min()
            assert self.full_range[1] >= y_true["EndTime"].max()
            assert self.full_range[0] <= min_y_pred_timestamp
            assert self.full_range[1] >= max_y_pred_timestamp
        for c, preds in y_pred.items():
            if y_pred[c][0, 0] > self.full_range[0]:
                y_pred[c] = np.array([np.array([self.full_range[0], y_pred[c][0, 1]]), *y_pred[c]])
            if y_pred[c][-1, 0] < self.full_range[1]:
                y_pred[c] = np.array([*y_pred[c], np.array([self.full_range[1], y_pred[c][-1, 1]])])

        events_pred_per_channel = dict()
        for channel_name, channel_pred in y_pred.items():
            events_pred_per_channel[channel_name] = convert_time_series_to_events(np.asarray(channel_pred))

        filtered_y_true = y_true.copy()
        for col, val in self.selected_labels.items():
            filtered_y_true = filtered_y_true[filtered_y_true[col].isin(val)]

        # fix for point anomalies
        point_anomalies = (filtered_y_true["StartTime"] == filtered_y_true["EndTime"])
        filtered_y_true.loc[point_anomalies, "EndTime"] = filtered_y_true.loc[point_anomalies, "StartTime"] + pd.Timedelta(milliseconds=1)

        unique_ids = filtered_y_true["ID"].unique()
        global_precisions = []
        global_recalls = []
        global_f_scores = []
        global_subsystem_precisions = []
        global_subsystem_recalls = []
        global_subsystem_f_scores = []

        aid_channels_intervals = dict()
        for aid in unique_ids:
            gt = filtered_y_true[filtered_y_true["ID"] == aid]

            channels_intervals = dict()
            for c in all_channels:
                c_gt = gt[gt["Channel"] == c]
                c_gt_intervals = []
                for _, row in c_gt[["StartTime", "EndTime"]].iterrows():
                    c_gt_intervals.append(P.closed(*row))
                channels_intervals[c] = P.Interval(*c_gt_intervals)
            aid_channels_intervals[aid] = channels_intervals


        for aid in unique_ids:
            channels_intervals = aid_channels_intervals[aid]

            full_interval = []
            for interval in channels_intervals.values():
                full_interval.append(interval)
            full_interval = P.Interval(*full_interval)

            true_positives = 0
            false_positives = 0
            false_negatives = 0
            for i, c in enumerate(all_channels):
                is_channel_affected = not channels_intervals[c].empty
                detection_interval = full_interval & events_pred_per_channel[c]
                is_channel_detected = not detection_interval.empty
                if is_channel_affected and is_channel_detected:
                    true_positives += 1
                elif is_channel_affected and not is_channel_detected:
                    false_negatives += 1
                elif not is_channel_affected and is_channel_detected:
                    # Remove any false detections that overlap with true positives for other anomalies
                    for id, a_intervals in aid_channels_intervals.items():
                        if aid == id:
                            continue
                        if a_intervals[c].empty:
                            continue
                        if not (detection_interval & a_intervals[c]).empty:
                            break
                    else:
                        false_positives += 1

            precision, recall, f_score = self.get_pr_re_f_score(true_positives, false_positives, false_negatives)

            global_precisions.append(precision)
            global_recalls.append(recall)
            global_f_scores.append(f_score)

            # Subsystem level
            if subsystems_mapping is not None:

                true_positives = 0
                false_positives = 0
                false_negatives = 0
                for sid, subsystem_channel_names in subsystems_mapping.items():
                    subsystem_channel_names = [c for c in subsystem_channel_names if c in all_channels]
                    if len(subsystem_channel_names) == 0:
                        continue

                    subsystem_interval = []
                    for channel_name in subsystem_channel_names:
                        subsystem_interval.append(channels_intervals[channel_name])
                    subsystem_interval = P.Interval(*subsystem_interval)

                    is_subsystem_affected = not subsystem_interval.empty
                    detected_intervals = {c: full_interval & events_pred_per_channel[c] for c in subsystem_channel_names}
                    is_subsystem_detected = np.any([not d_i.empty for d_i in detected_intervals.values()])

                    if is_subsystem_affected and is_subsystem_detected:
                        true_positives += 1
                    elif is_subsystem_affected and not is_subsystem_detected:
                        false_negatives += 1
                    elif not is_subsystem_affected and is_subsystem_detected:
                        # Remove any false detections that overlap with true positives for other anomalies
                        for id, a_intervals in aid_channels_intervals.items():
                            if aid == id:
                                continue
                            for d_c, d_interval in detected_intervals.items():
                                if a_intervals[d_c].empty:
                                    continue
                                if not (d_interval & a_intervals[d_c]).empty:
                                    detected_intervals[d_c] = P.empty()

                        # And check again after removal
                        is_subsystem_detected = np.any([not d_i.empty for d_i in detected_intervals.values()])
                        if is_subsystem_detected:
                            false_positives += 1

                precision, recall, f_score = self.get_pr_re_f_score(true_positives, false_positives, false_negatives)

                global_subsystem_precisions.append(precision)
                global_subsystem_recalls.append(recall)
                global_subsystem_f_scores.append(f_score)

        # print(global_f_scores)
        # print(global_subsystem_f_scores)

        result_dict = {"channel_precision": np.mean(global_precisions),
                       "channel_recall": np.mean(global_recalls),
                       f"channel_F{self._beta:.2f}": np.mean(global_f_scores)}

        if subsystems_mapping is not None:
            result_dict.update({"subsystem_precision": np.mean(global_subsystem_precisions),
                                "subsystem_recall": np.mean(global_subsystem_recalls),
                                f"subsystem_F{self._beta:.2f}": np.mean(global_subsystem_f_scores)})

        return result_dict