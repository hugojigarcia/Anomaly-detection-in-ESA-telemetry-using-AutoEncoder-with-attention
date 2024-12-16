import pandas as pd
from metrics_libraries.basic_methods import precision_corrected_score, recall_score, f05_score
from datetime import datetime


def calculate_event_wise_metrics(grouped_anomalies: pd.DataFrame,
                                 true_anomalies: pd.DataFrame,
                                 start_date: datetime,
                                 end_date: datetime,
                                 unit_to_consider: str = 'S'):
    event_wise_basic_metrics = calculate_event_wise_basic_metrics(grouped_anomalies, true_anomalies, unit_to_consider)
    event_wise_tp, event_wise_fp, event_wise_fn = event_wise_basic_metrics["True Positives"], event_wise_basic_metrics["False Positives"], event_wise_basic_metrics["False Negatives"]
    tnrt, tnt, nt = calculate_TNRt(grouped_anomalies, true_anomalies, start_date, end_date)
    event_wise_precision = precision_corrected_score(event_wise_tp, event_wise_fp, tnrt)
    event_wise_recall = recall_score(event_wise_tp, event_wise_fn)
    return {
        "tp": event_wise_tp,
        "fp": event_wise_fp,
        "fn": event_wise_fn,
        "tnrt": tnrt,
        "tnt": tnt,
        "nt": nt,
        "event_wise_precision": event_wise_precision,
        "event_wise_recall": event_wise_recall,
        "event_wise_f05": f05_score(event_wise_precision, event_wise_recall)
    }


def calculate_event_wise_basic_metrics(grouped_anomalies: pd.DataFrame,
                                       true_anomalies: pd.DataFrame,
                                       unit_to_consider: str = 'S'):
    grouped_anomalies['time_start_f'] = pd.to_datetime(grouped_anomalies['time_start']).dt.floor(unit_to_consider)
    grouped_anomalies['time_end_f'] = pd.to_datetime(grouped_anomalies['time_end']).dt.floor(unit_to_consider)
    true_anomalies = true_anomalies.assign(StartTime_f = pd.to_datetime(true_anomalies['StartTime']).dt.floor(unit_to_consider))
    true_anomalies = true_anomalies.assign(EndTime_f = pd.to_datetime(true_anomalies['EndTime']).dt.floor(unit_to_consider))

    def __calculate_true_anomalie_coverage(row):
        total_coverage = 0
        for _, group_row in grouped_anomalies.iterrows():
            # Calculate the overlap start and end times
            overlap_start = max(row['StartTime_f'], group_row['time_start'])
            overlap_end = min(row['EndTime_f'], group_row['time_end'])
            # Calculate the overlap duration in seconds
            overlap_duration = max(0, (overlap_end - overlap_start).total_seconds())
            # Calculate the duration of the true_anomalie
            true_anomalie_duration = (row['EndTime_f'] - row['StartTime_f']).total_seconds()
            # Add the percentage of this specific overlap to the total coverage
            if true_anomalie_duration > 0:
                total_coverage += (overlap_duration / true_anomalie_duration) * 100
        # Ensure the percentage doesn't exceed 100%
        return min(total_coverage, 100)
    
    def __calculate_overlap(row_start, row_end, StartTime_f, EndTime_f):
        """Función para calcular el porcentaje de solapamiento."""
        overlap_start = max(row_start, StartTime_f)
        overlap_end = min(row_end, EndTime_f)
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        
        # Duración total de la anomalía en grouped_anomalies
        row_duration = (row_end - row_start).total_seconds()
        
        # Calcular el porcentaje de solapamiento
        if row_duration > 0:
            overlap_percentage = round((overlap_duration / row_duration) * 100, 2)
        elif overlap_duration == 0:
            overlap_percentage = 100
        else:
            overlap_percentage = 0
        return overlap_percentage

    def __find_matching_time(row):
        """Función para encontrar la coincidencia y calcular el porcentaje de solapamiento."""
        for i, target_row in true_anomalies.iterrows():
            if (row['time_start_f'] <= target_row['EndTime_f']) and (row['time_end_f'] >= target_row['StartTime_f']):
                overlap_percentage = __calculate_overlap(
                    row['time_start_f'], row['time_end_f'], 
                    target_row['StartTime_f'], target_row['EndTime_f']
                )
                return target_row['StartTime_f'], target_row['EndTime_f'], overlap_percentage
        return "", "", 0  # Si no hay coincidencia, devolver "", "", 0
    
    
    
    
    if len(true_anomalies) > 0:
        true_anomalies['coverage_percentage'] = true_anomalies.apply(__calculate_true_anomalie_coverage, axis=1)
    else:
        true_anomalies['coverage_percentage'] = []
    if len(grouped_anomalies) > 0:
        grouped_anomalies[['true_anomalie_start', 'true_anomalie_end', 'overlap_percentage']] = pd.DataFrame(
            grouped_anomalies.apply(__find_matching_time, axis=1).tolist(), index=grouped_anomalies.index
        )
    else:
        grouped_anomalies[['true_anomalie_start', 'true_anomalie_end', 'overlap_percentage']] = None

    # TPe = (true_anomalies['coverage_percentage'] > 0.0).sum()
    TPe = len(true_anomalies[true_anomalies['coverage_percentage'] > 0.0]['ID'].unique())
    FPe = len(grouped_anomalies) - (grouped_anomalies['overlap_percentage'] > 0.0).sum()
    # FNe = len(true_anomalies) - TPe
    FNe = len(true_anomalies['ID'].unique()) - TPe


    return {
        "True Positives": TPe,
        "False Positives": FPe,
        "False Negatives": FNe
    }


def calculate_TNRt(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame, start_date: datetime, end_date: datetime):
    TNt = __calculate_TNt(grouped_anomalies, true_anomalies, start_date, end_date)
    Nt = __calculate_Nt(true_anomalies, start_date, end_date)
    return TNt / Nt, TNt, Nt

def __calculate_TNt(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame, start_date: datetime, end_date: datetime):
    # Función para fusionar intervalos superpuestos
    def merge_intervals(intervals):
        if not intervals:
            return []
        
        # Ordenar los intervalos por el tiempo de inicio
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]

        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]

            if current_start <= last_end:
                # Si los intervalos se solapan, fusionarlos
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # Si no se solapan, agregar un nuevo intervalo
                merged.append((current_start, current_end))

        return merged

    # Combinar los rangos de ambos dataframes en una lista
    intervals = []

    # Añadir rangos del dataframe true_anomalies
    for _, row in true_anomalies.iterrows():
        intervals.append((row['StartTime'], row['EndTime']))

    # Añadir rangos del dataframe grouped_anomalies
    for _, row in grouped_anomalies.iterrows():
        intervals.append((row['time_start'], row['time_end']))

    # Fusionar los intervalos superpuestos
    merged_intervals = merge_intervals(intervals)

    # Ahora, vamos a calcular el tiempo total no cubierto por los intervalos
    total_time_ms = (end_date - start_date).total_seconds() * 1000

    # Restar el tiempo cubierto por los intervalos fusionados
    covered_time_ms = 0
    for start, end in merged_intervals:
        if start < end_date and end > start_date:
            # Asegurarse de que el intervalo esté dentro del rango total
            covered_start = max(start, start_date)
            covered_end = min(end, end_date)
            covered_time_ms += (covered_end - covered_start).total_seconds() * 1000

    # Calcular TNt
    TNt = total_time_ms - covered_time_ms
    return TNt

def __calculate_Nt(true_anomalies: pd.DataFrame, start_date: datetime, end_date: datetime):
    if len(true_anomalies) > 0:
        # Sort intervals by StartTime and EndTime to help with merging overlaps
        _true_anomalies_sorted = true_anomalies.sort_values(by=['StartTime', 'EndTime'])

        # Initialize a list to store merged intervals
        merged_intervals = []

        # Initialize the first interval
        current_start = _true_anomalies_sorted.iloc[0]['StartTime']
        current_end = _true_anomalies_sorted.iloc[0]['EndTime']

        for i in range(1, len(_true_anomalies_sorted)):
            row = _true_anomalies_sorted.iloc[i]
            start = row['StartTime']
            end = row['EndTime']
            
            # Check if there is an overlap with the current interval
            if start <= current_end:  # If there's overlap, merge the intervals
                current_end = max(current_end, end)
            else:  # If there's no overlap, save the current interval and start a new one
                merged_intervals.append((current_start, current_end))
                current_start = start
                current_end = end

        # Add the last interval
        merged_intervals.append((current_start, current_end))

        # Calculate duration for each merged interval and the sum
        # total_esa_anomalies_duration = sum([(end - start).total_seconds() * 1000 for start, end in merged_intervals])
        total_esa_anomalies_duration = 0
        for start, end in merged_intervals:
            if start < end_date and end > start_date:
                # Asegurarse de que el intervalo esté dentro del rango total
                covered_start = max(start, start_date)
                covered_end = min(end, end_date)
                total_esa_anomalies_duration += (covered_end - covered_start).total_seconds() * 1000

    else:
        total_esa_anomalies_duration = 0
    total_time = (end_date - start_date).total_seconds() * 1000
    Nt = total_time - total_esa_anomalies_duration
    return Nt