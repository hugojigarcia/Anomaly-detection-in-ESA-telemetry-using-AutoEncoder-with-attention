import pandas as pd
from datetime import datetime

from metrics_libraries.event_wise import calculate_event_wise_metrics
from metrics_libraries.subsystem_aware import calculate_subsystem_aware_metrics
from metrics_libraries.channel_aware import calculate_channel_aware_metrics
from metrics_libraries.alarming_precision import calculate_alarming_precision_metrics
from metrics_libraries.adtqc import calculate_adtqc_metrics
from metrics_libraries.affiliation_based import calculate_affiliation_based_metrics

#%%
def calculate_esa_metrics(grouped_anomalies: pd.DataFrame,
                          true_anomalies: pd.DataFrame,
                          channel_list_path: str,
                          start_date: datetime,
                          end_date: datetime,
                          influence_limit: float = 0.0):
    event_wise_metrics = calculate_event_wise_metrics(grouped_anomalies, true_anomalies, start_date, end_date)
    _subsystems_mapping = pd.read_csv(channel_list_path).groupby('Subsystem')['Channel'].apply(list).to_dict()
    subsystem_aware_metrics = calculate_subsystem_aware_metrics(grouped_anomalies, true_anomalies, _subsystems_mapping)
    channel_aware_metrics = calculate_channel_aware_metrics(grouped_anomalies, true_anomalies, influence_limit)
    alarming_precision_metrics = calculate_alarming_precision_metrics(grouped_anomalies, true_anomalies)
    adtqc_metrics = calculate_adtqc_metrics(grouped_anomalies, true_anomalies)
    affiliation_based_metrics = calculate_affiliation_based_metrics(grouped_anomalies, true_anomalies)
    return {
         **event_wise_metrics,
         **subsystem_aware_metrics,
         **channel_aware_metrics,
         **alarming_precision_metrics,
         **adtqc_metrics,
         **affiliation_based_metrics
    }



#%%
def print_esa_metrics_table(metrics: dict):
    print(f"|-----------------------------------------|")
    print(f"|     Event-wise    |  Precision  | {metrics['event_wise_precision']:.3f} |")
    print(f"|     Event-wise    |   Recall    | {metrics['event_wise_recall']:.3f} |")
    print(f"|     Event-wise    |    F0.5     | {metrics['event_wise_f05']:.3f} |")
    print(f"|-----------------------------------------|")
    print(f"|  Subsystem-aware  |  Precision  | {metrics['subsystem_aware_precision']:.3f} |")
    print(f"|  Subsystem-aware  |   Recall    | {metrics['subsystem_aware_recall']:.3f} |")
    print(f"|  Subsystem-aware  |    F0.5     | {metrics['subsystem_aware_f05']:.3f} |")
    print(f"|-----------------------------------------|")
    print(f"|   Channel-aware   |  Precision  | {metrics['channel_aware_precision']:.3f} |")
    print(f"|   Channel-aware   |   Recall    | {metrics['channel_aware_recall']:.3f} |")
    print(f"|   Channel-aware   |    F0.5     | {metrics['channel_aware_f05']:.3f} |")
    print(f"|-----------------------------------------|")
    print(f"|       Alarming precision        | {metrics['alarming_precision']:.3f} |")
    print(f"|-----------------------------------------|")
    print(f"|       ADTQC       | After ratio | {metrics['adtqc_after_ratio']:.3f} |")
    print(f"|       ADTQC       |    Score    | {metrics['adtqc_score']:.3f} |")
    print(f"|-----------------------------------------|")
    print(f"| Affiliation-based |  Precision  | {metrics['affiliation_based_precision']:.3f} |")
    print(f"| Affiliation-based |   Recall    | {metrics['affiliation_based_recall']:.3f} |")
    print(f"| Affiliation-based |    F0.5     | {metrics['affiliation_based_f05']:.3f} |")
    print(f"|-----------------------------------------|")