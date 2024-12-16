import pandas as pd
from metrics_libraries.basic_methods import process_pred_channels
from metrics_libraries.basic_methods import precision_score, recall_score, f05_score

def calculate_channel_aware_metrics(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame, influence_limit: float = 0.0):
    channel_aware_basic_metrics = calculate_channel_aware_basic_metrics(grouped_anomalies, true_anomalies, influence_limit)
    channel_aware_tp, channel_aware_fp, channel_aware_fn = channel_aware_basic_metrics["True Positives"], channel_aware_basic_metrics["False Positives"], channel_aware_basic_metrics["False Negatives"]
    channel_aware_precision = precision_score(channel_aware_tp, channel_aware_fp)
    channel_aware_recall = recall_score(channel_aware_tp, channel_aware_fn)
    return {
        "channel_aware_precision": channel_aware_precision,
        "channel_aware_recall": channel_aware_recall,
        "channel_aware_f05": f05_score(channel_aware_precision, channel_aware_recall)
    }


def calculate_channel_aware_basic_metrics(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame, influence_limit: float = 0.0):
    """
    Calcula True Positives (TP), False Positives (FP) y False Negatives (FN) basados en los dataframes grouped_anomalies y true_anomalies.
    
    Parameters:
    ----------
    grouped_anomalies : pd.DataFrame
        DataFrame de predicciones (anomalías agrupadas por rango de tiempo y canales)
    true_anomalies : pd.DataFrame
        DataFrame de anomalías verdaderas (ground truth) con canales afectados y rangos de tiempo
    
    Returns:
    ----------
    metrics : dict
        Diccionario con TP, FP y FN.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Convertir las columnas de tiempo a formato datetime (si no lo están ya)
    grouped_anomalies['time_start'] = pd.to_datetime(grouped_anomalies['time_start'])
    grouped_anomalies['time_end'] = pd.to_datetime(grouped_anomalies['time_end'])
    true_anomalies['StartTime'] = pd.to_datetime(true_anomalies['StartTime'])
    true_anomalies['EndTime'] = pd.to_datetime(true_anomalies['EndTime'])

    # Iterar sobre cada predicción en grouped_anomalies
    for _, pred_row in grouped_anomalies.iterrows():
        pred_channels = pred_row['Ranking_cols']  # Canales predichos
        pred_start = pred_row['time_start']
        pred_end = pred_row['time_end']
        matched = False

        # Desglosar los canales en una lista (pueden estar en forma de string con porcentajes)
        # pred_channels = [ch.split(' ')[0] for ch in raw_pred_channels.strip('[]').split(',')]
        pred_channels = process_pred_channels(pred_channels, influence_limit)
        

        # Iterar sobre cada fila de true_anomalies para comparar
        for _, true_row in true_anomalies.iterrows():
            true_channel = true_row['Channel']
            true_start = true_row['StartTime']
            true_end = true_row['EndTime']

            # Verificar si el canal está en la lista de canales predichos
            if true_channel in pred_channels:
                # Verificar si hay solapamiento de tiempos
                if (pred_start <= true_end) and (pred_end >= true_start):
                    true_positives += 1
                    matched = True
                    break

        if not matched:
            false_positives += 1

    # Calcular los false negatives (anomalías verdaderas no detectadas)
    for _, true_row in true_anomalies.iterrows():
        true_channel = true_row['Channel']
        true_start = true_row['StartTime']
        true_end = true_row['EndTime']
        matched = False

        # Verificar si esta anomalía real fue detectada por alguna predicción
        for _, pred_row in grouped_anomalies.iterrows():
            # pred_channels = [ch.split(' ')[0] for ch in pred_row['Ranking_cols'].strip('[]').split(',')]
            pred_channels = process_pred_channels(pred_row['Ranking_cols'])
            pred_start = pred_row['time_start']
            pred_end = pred_row['time_end']

            if true_channel in pred_channels and (pred_start <= true_end) and (pred_end >= true_start):
                matched = True
                break

        if not matched:
            false_negatives += 1

    return {
        "True Positives": true_positives,
        "False Positives": false_positives,
        "False Negatives": false_negatives
    }




#%%
# result = []
# for i, row_grouped in grouped_anomalies.iterrows():
#     start_grouped, end_grouped = row_grouped["time_start"], row_grouped["time_end"]
#     ranking = []
#     for el in row["Ranking_cols"]:
#         _aux = el.split(" ")
#         channel, influnce = _aux[0], float(_aux[1].strip("()%"))
#         if influnce > influnce_limit:
#             ranking.append(channel)

#     print(ranking)
#     true_positives = set()
#     false_negatives = set()
#     for j, row_true in true_anomalies.iterrows():
#         channel_true, start_true, end_true = row_true["Channel"], row_true["StartTime"], row_true["EndTime"]
#         if channel_true in ranking and ranges_overlap(start_grouped, end_grouped, start_true, end_true):
#             true_positives.add(channel_true)
#         else:
#             false_negatives.add(channel_true)
#     true_positives = list(true_positives)
#     false_positives = [el for el in ranking if el not in true_positives]
#     false_negatives = list(false_negatives)
#     print("true_positives", len(true_positives), true_positives)
#     print("false_positives", len(false_positives), false_positives)
#     print("false_negatives", len(false_negatives), false_negatives)
#     print()