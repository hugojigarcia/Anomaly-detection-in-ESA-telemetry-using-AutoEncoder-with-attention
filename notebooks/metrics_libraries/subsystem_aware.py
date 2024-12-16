import pandas as pd
from collections import defaultdict
from metrics_libraries.basic_methods import precision_score, recall_score, f05_score

### SUBSYSTEM AWARE
#%%
def calculate_subsystem_aware_metrics(grouped_anomalies, true_anomalies, subsystems_mapping):
    subsystem_aware_basic_metrics = calculate_subsystem_aware_basic_metrics(grouped_anomalies, true_anomalies, subsystems_mapping)
    subsystem_aware_tp, subsystem_aware_fp, subsystem_aware_fn = subsystem_aware_basic_metrics["True Positives"], subsystem_aware_basic_metrics["False Positives"], subsystem_aware_basic_metrics["False Negatives"]
    subsystem_aware_precision = precision_score(subsystem_aware_tp, subsystem_aware_fp)
    subsystem_aware_recall = recall_score(subsystem_aware_tp, subsystem_aware_fn)
    return {
        "subsystem_aware_precision": subsystem_aware_precision,
        "subsystem_aware_recall": subsystem_aware_recall,
        "subsystem_aware_f05": f05_score(subsystem_aware_precision, subsystem_aware_recall)
    }


#%%
def calculate_subsystem_aware_basic_metrics(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame, subsystems_mapping: dict) -> dict:
    """
    Calculate subsystem-aware true positives, false positives, and false negatives.

    Parameters:
    -----------
    grouped_anomalies: pd.DataFrame
        The predicted anomalies dataframe with channels in 'Ranking_cols'.
    true_anomalies: pd.DataFrame
        The true anomalies dataframe with channels and associated information.
    subsystems_mapping: dict
        A dictionary mapping subsystems to channels (subsystem_name -> list of channel_names).
    
    Returns:
    --------
    dict: Subsystem-aware true positives, false positives, and false negatives.
    """
    # Initialize counters
    subsystem_tp = 0
    subsystem_fp = 0
    subsystem_fn = 0

    # Create reverse mapping of channels to subsystems
    channel_to_subsystem = {}
    for subsystem, channels in subsystems_mapping.items():
        for channel in channels:
            channel_to_subsystem[channel] = subsystem

    # Group true anomalies by subsystems
    subsystem_true_anomalies = defaultdict(list)
    for _, row in true_anomalies.iterrows():
        channel = row['Channel']
        if channel in channel_to_subsystem:
            subsystem = channel_to_subsystem[channel]
            subsystem_true_anomalies[subsystem].append(row)

    # Track which true anomalies have been detected
    detected_true_anomalies = set()

    # Group predicted anomalies by subsystems
    for _, row in grouped_anomalies.iterrows():
        predicted_channels = [col.split(' ')[0] for col in row['Ranking_cols']]  # Extract channel names from ranking cols
        predicted_subsystems = set([channel_to_subsystem[ch] for ch in predicted_channels if ch in channel_to_subsystem])
        
        matched_subsystem = False
        for subsystem in predicted_subsystems:
            true_anomalies_in_subsystem = subsystem_true_anomalies[subsystem]
            
            for true_anomaly in true_anomalies_in_subsystem:
                if row['time_start'] <= true_anomaly['EndTime'] and row['time_end'] >= true_anomaly['StartTime']:
                    # Overlap found -> True Positive
                    subsystem_tp += 1
                    detected_true_anomalies.add((true_anomaly['ID'], subsystem))
                    matched_subsystem = True
                    break

        if not matched_subsystem:
            # No true anomaly found in the predicted subsystem -> False Positive
            subsystem_fp += 1

    # Calculate false negatives
    for subsystem, anomalies in subsystem_true_anomalies.items():
        for anomaly in anomalies:
            if (anomaly['ID'], subsystem) not in detected_true_anomalies:
                # True anomaly not detected -> False Negative
                subsystem_fn += 1

    return {
        'True Positives': subsystem_tp,
        'False Positives': subsystem_fp,
        'False Negatives': subsystem_fn
    }