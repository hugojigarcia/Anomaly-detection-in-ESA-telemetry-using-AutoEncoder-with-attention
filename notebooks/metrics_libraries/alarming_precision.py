import pandas as pd
import numpy as np
from intervaltree import Interval, IntervalTree

def calculate_alarming_precision_metrics(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame) -> float:
    """
    Calculate alarming precision based on grouped_anomalies (predictions) and true_anomalies (ground truth).
    
    :param grouped_anomalies: DataFrame of predicted anomalies with columns ["time_start", "time_end", "is_anomaly"]
    :param true_anomalies: DataFrame of true anomalies with columns ["StartTime", "EndTime"]
    :return: alarming precision score
    """
    
    # Create intervals for true anomalies
    true_anomaly_intervals = IntervalTree()
    for _, row in true_anomalies.iterrows():
        interval_start, interval_end = pd.to_datetime(row['StartTime']).timestamp(), pd.to_datetime(row['EndTime']).timestamp()
        if interval_start == interval_end:
            interval_end += 1e-6
        true_anomaly_intervals.add(Interval(interval_start, interval_end))
    true_positives = 0
    false_positives = 0
    redundant_detections = 0

    # Iterate over grouped anomalies (predictions)
    for _, pred in grouped_anomalies.iterrows():
        pred_start = pd.to_datetime(pred['time_start']).timestamp()
        pred_end = pd.to_datetime(pred['time_end']).timestamp()
        # pred_interval = Interval(pred_start, pred_end)

        # Check if the predicted anomaly overlaps with any true anomaly
        overlapping_intervals = true_anomaly_intervals.overlap(pred_start, pred_end)
        
        if overlapping_intervals:
            true_positives += 1
            # Count redundant detections (overlapping multiple times)
            redundant_detections += len(overlapping_intervals) - 1
        else:
            false_positives += 1

    # Calculate alarming precision
    divider = true_positives + redundant_detections
    if divider == 0:
        return {"alarming_precision": 0.0}
    else:
        return {"alarming_precision": true_positives / divider}