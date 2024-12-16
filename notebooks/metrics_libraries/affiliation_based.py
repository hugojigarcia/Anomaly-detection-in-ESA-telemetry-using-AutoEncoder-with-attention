import pandas as pd

def calculate_affiliation_based_metrics(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame):
    return {
        "affiliation_based_precision": 0,
        "affiliation_based_recall": 0,
        "affiliation_based_f05": 0
    }



#%%
# import pandas as pd
# import numpy as np
# from intervaltree import Interval, IntervalTree

# def calculate_aff_precision_recall(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame) -> dict:
#     """
#     Calculate AFF_precision and AFF_recall based on predicted and true anomalies.
    
#     :param grouped_anomalies: DataFrame representing predicted anomalies.
#     :param true_anomalies: DataFrame representing true anomalies.
#     :return: Dictionary with AFF_precision and AFF_recall.
#     """
    
#     def convert_to_intervals(df, start_col, end_col):
#         """
#         Convert start and end times to a list of Interval objects.
#         """
#         return [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in zip(df[start_col], df[end_col])]

#     # Convert the predictions and true anomalies to intervals
#     pred_intervals = convert_to_intervals(grouped_anomalies, 'time_start', 'time_end')
#     true_intervals = convert_to_intervals(true_anomalies, 'StartTime', 'EndTime')

#     # Create IntervalTrees for quick overlap checking
#     pred_tree = IntervalTree([Interval(start.value, end.value, 'pred') for start, end in pred_intervals])
#     true_tree = IntervalTree([Interval(start.value, end.value, 'true') for start, end in true_intervals])
    
#     # Variables to track precision and recall
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0

#     # Check each predicted interval against the true intervals
#     for pred_interval in pred_intervals:
#         overlap = len(true_tree.overlap(pred_interval[0].value, pred_interval[1].value)) > 0
#         if overlap:
#             true_positives += 1
#         else:
#             false_positives += 1

#     # Check if all true intervals are detected
#     for true_interval in true_intervals:
#         overlap = len(pred_tree.overlap(true_interval[0].value, true_interval[1].value)) > 0
#         if not overlap:
#             false_negatives += 1

#     # Calculate AFF_precision and AFF_recall
#     divider = true_positives + false_positives
#     if divider == 0:
#         aff_precision = 0.0
#     else:
#         aff_precision = true_positives / divider

#     divider = true_positives + false_negatives
#     if divider == 0:
#         aff_recall = 0.0
#     else:
#         aff_recall = true_positives / divider

#     return {
#         'AFF_precision': aff_precision,
#         'AFF_recall': aff_recall
#     }

# # Ejemplo de uso
# result = calculate_aff_precision_recall(grouped_anomalies, true_anomalies)
# print(result)


#%%
# import pandas as pd
# import numpy as np
# from intervaltree import Interval, IntervalTree

# NANOSECONDS_IN_SECOND = 1_000_000_000

# def calculate_aff_precision_recall(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame, selected_labels=None) -> dict:
#     """
#     Calcula AFF_precision y AFF_recall basados en anomalías predichas y verdaderas.
    
#     :param grouped_anomalies: DataFrame con las anomalías predichas.
#     :param true_anomalies: DataFrame con las anomalías verdaderas.
#     :param selected_labels: Diccionario para filtrar anomalías verdaderas, e.g., {'Category': ['Anomaly']}
#     :return: Diccionario con AFF_precision y AFF_recall.
#     """
    
#     def convert_to_intervals(df, start_col, end_col):
#         """
#         Convierte columnas de inicio y fin a una lista de tuplas con timestamps en nanosegundos.
#         """
#         return [(pd.Timestamp(start).value, pd.Timestamp(end).value) for start, end in zip(df[start_col], df[end_col])]
    
#     def subtract_intervals(full_range, intervals):
#         """
#         Resta una lista de intervalos del rango completo.
        
#         :param full_range: Tupla (inicio, fin)
#         :param intervals: Lista de tuplas (inicio, fin)
#         :return: Lista de tuplas (inicio, fin) representando los intervalos nominales
#         """
#         sorted_intervals = sorted(intervals, key=lambda x: x[0])
#         result = []
#         current_start = full_range[0]
#         for interval in sorted_intervals:
#             if interval[0] > current_start:
#                 result.append((current_start, interval[0]))
#             current_start = max(current_start, interval[1])
#         if current_start < full_range[1]:
#             result.append((current_start, full_range[1]))
#         return result
    
#     # Filtrar anomalías verdaderas si se proporcionan etiquetas seleccionadas
#     if selected_labels:
#         filtered_y_true = true_anomalies.copy()
#         for col, val in selected_labels.items():
#             filtered_y_true = filtered_y_true[filtered_y_true[col].isin(val)]
#     else:
#         filtered_y_true = true_anomalies.copy()
    
#     # Convertir predicciones y anomalías verdaderas a intervalos
#     pred_intervals = convert_to_intervals(grouped_anomalies, 'time_start', 'time_end')
#     true_intervals = convert_to_intervals(filtered_y_true, 'StartTime', 'EndTime')
    
#     # Definir el rango completo
#     min_start = min(
#         filtered_y_true["StartTime"].min(),
#         grouped_anomalies["time_start"].min()
#     )
#     max_end = max(
#         filtered_y_true["EndTime"].max(),
#         grouped_anomalies["time_end"].max()
#     )
#     full_range = (pd.Timestamp(min_start).value, pd.Timestamp(max_end).value)
    
#     # Crear IntervalTrees para búsquedas rápidas
#     true_tree = IntervalTree([Interval(start, end) for start, end in true_intervals])
#     pred_tree = IntervalTree([Interval(start, end) for start, end in pred_intervals])
    
#     # Inicializar contadores
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
    
#     # Para rastrear anomalías verdaderas detectadas
#     detected_true_ids = set()
    
#     # Calcular True Positives y False Positives
#     for pred_start, pred_end in pred_intervals:
#         overlaps = true_tree.overlap(pred_start, pred_end)
#         if overlaps:
#             true_positives += 1
#             # Opcional: Rastrear IDs detectados si están disponibles
#             # for interval in overlaps:
#             #     detected_true_ids.add(interval.data)
#         else:
#             false_positives += 1
    
#     # Calcular False Negatives
#     false_negatives = len(true_intervals) - true_positives  # Simplificación
    
#     # Calcular precisión y recall Event-Wise
#     if true_positives + false_positives == 0:
#         ew_precision = 0.0
#     else:
#         ew_precision = true_positives / (true_positives + false_positives)
    
#     if true_positives + false_negatives == 0:
#         ew_recall = 0.0
#     else:
#         ew_recall = true_positives / (true_positives + false_negatives)
    
#     # Calcular intervalos nominales (sin anomalías verdaderas)
#     nominal_intervals = subtract_intervals(full_range, true_intervals)
    
#     # Calcular tiempo nominal total en segundos
#     nominal_seconds = sum((end - start) / NANOSECONDS_IN_SECOND for start, end in nominal_intervals)
    
#     # Calcular tiempo de falsos positivos en segundos
#     false_positive_seconds = 0
#     for pred_start, pred_end in pred_intervals:
#         for nom_start, nom_end in nominal_intervals:
#             overlap_start = max(pred_start, nom_start)
#             overlap_end = min(pred_end, nom_end)
#             if overlap_start < overlap_end:
#                 false_positive_seconds += (overlap_end - overlap_start) / NANOSECONDS_IN_SECOND
    
#     # Calcular TNR (True Negative Rate)
#     if nominal_seconds > 0:
#         tnr = 1 - (false_positive_seconds / nominal_seconds)
#     else:
#         tnr = 1.0  # Evitar división por cero
    
#     # Ajustar precisión para obtener AFF_precision
#     aff_precision = ew_precision * tnr
    
#     # AFF_recall es igual a ew_recall
#     aff_recall = ew_recall
    
#     return {
#         'AFF_precision': aff_precision,
#         'AFF_recall': aff_recall
#     }

# # Calcular AFF_precision y AFF_recall
# result = calculate_aff_precision_recall(grouped_anomalies, true_anomalies, selected_labels={'Category': ['Anomaly']})
# print(result)


#%%
# import pandas as pd
# import numpy as np
# from collections import defaultdict

# def calculate_aff_precision_recall(grouped_anomalies: pd.DataFrame, true_anomalies: pd.DataFrame):
#     # Convertir las columnas de tiempo a datetime
#     grouped_anomalies['time_start'] = pd.to_datetime(grouped_anomalies['time_start'])
#     grouped_anomalies['time_end'] = pd.to_datetime(grouped_anomalies['time_end'])
#     true_anomalies['StartTime'] = pd.to_datetime(true_anomalies['StartTime'])
#     true_anomalies['EndTime'] = pd.to_datetime(true_anomalies['EndTime'])

#     # Inicializar diccionarios para precisión y recall
#     precision_dict = defaultdict(list)
#     recall_dict = defaultdict(list)

#     # Iterar sobre las anomalías agrupadas y las verdaderas
#     for _, pred_row in grouped_anomalies.iterrows():
#         # Crear un intervalo de tiempo para la predicción
#         predicted_zone = pd.Interval(pred_row['time_start'], pred_row['time_end'])

#         # Filtrar las verdaderas anomalías que se superponen con la predicción
#         overlaps = true_anomalies[
#             (true_anomalies['StartTime'] < pred_row['time_end']) &
#             (true_anomalies['EndTime'] > pred_row['time_start'])
#         ]

#         # Contar el número de anomalías verdaderas en la zona de predicción
#         if not overlaps.empty:
#             # Calcular precisión y recall
#             tp = len(overlaps)  # Verdaderos positivos
#             fp = 1  # Falsos positivos: aquí asumimos que cada predicción es un falso positivo si hay overlap
#             fn = len(true_anomalies) - tp  # Falsos negativos

#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0

#             precision_dict[pred_row['ID']] = precision
#             recall_dict[pred_row['ID']] = recall

#     # Calcular promedios de precisión y recall
#     precision_list = list(precision_dict.values())
#     recall_list = list(recall_dict.values())

#     # Evitar divisiones por cero
#     if precision_list:
#         avg_precision = np.mean(precision_list)
#     else:
#         avg_precision = 0

#     if recall_list:
#         avg_recall = np.mean(recall_list)
#     else:
#         avg_recall = 0

#     return avg_precision, avg_recall

# precision, recall = calculate_aff_precision_recall(grouped_anomalies, true_anomalies)
# print("AFF Precision:", precision)
# print("AFF Recall:", recall)


# %%
# import math

# def interval_intersection(I = (1, 3), J = (2, 4)): 
#     """
#     Intersection between two intervals I and J
#     I and J should be either empty or represent a positive interval (no point)
    
#     :param I: an interval represented by start and stop
#     :param J: a second interval of the same form
#     :return: an interval representing the start and stop of the intersection (or None if empty)
#     """
#     if I is None:
#         return(None)
#     if J is None:
#         return(None)
        
#     I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
#     if I_inter_J[0] >= I_inter_J[1]:
#         return(None)
#     else:
#         return(I_inter_J)
    
# def affiliation_partition(Is = [(1,1.5),(2,5),(5,6),(8,9)], E_gt = [(1,2.5),(2.5,4.5),(4.5,10)]):
#     """
#     Cut the events into the affiliation zones
#     The presentation given here is from the ground truth point of view,
#     but it is also used in the reversed direction in the main function.
    
#     :param Is: events as a list of couples
#     :param E_gt: range of the affiliation zones
#     :return: a list of list of intervals (each interval represented by either 
#     a couple or None for empty interval). The outer list is indexed by each
#     affiliation zone of `E_gt`. The inner list is indexed by the events of `Is`.
#     """
#     out = [None] * len(E_gt)
#     for j in range(len(E_gt)):
#         E_gt_j = E_gt[j]
#         discarded_idx_before = [I[1] < E_gt_j[0] for I in Is]  # end point of predicted I is before the begin of E
#         discarded_idx_after = [I[0] > E_gt_j[1] for I in Is] # start of predicted I is after the end of E
#         kept_index = [not(a or b) for a, b in zip(discarded_idx_before, discarded_idx_after)]
#         Is_j = [x for x, y in zip(Is, kept_index) if y]
#         out[j] = [interval_intersection(I, E_gt[j]) for I in Is_j]
#     return(out)

# def test_events(events):
#     """
#     Verify the validity of the input events
#     :param events: list of events, each represented by a couple (start, stop)
#     :return: None. Raise an error for incorrect formed or non ordered events
#     """
#     if type(events) is not list:
#         raise TypeError('Input `events` should be a list of couples')
#     if not all([type(x) is tuple for x in events]):
#         raise TypeError('Input `events` should be a list of tuples')
#     if not all([len(x) == 2 for x in events]):
#         raise ValueError('Input `events` should be a list of couples (start, stop)')
#     if not all([x[0] <= x[1] for x in events]):
#         raise ValueError('Input `events` should be a list of couples (start, stop) with start <= stop')
#     if not all([events[i][1] < events[i+1][0] for i in range(len(events) - 1)]):
#         raise ValueError('Couples of input `events` should be disjoint and ordered')

# def infer_Trange(events_pred, events_gt):
#     """
#     Given the list of events events_pred and events_gt, get the
#     smallest possible Trange corresponding to the start and stop indexes 
#     of the whole series.
#     Trange will not influence the measure of distances, but will impact the
#     measures of probabilities.
    
#     :param events_pred: a list of couples corresponding to predicted events
#     :param events_gt: a list of couples corresponding to ground truth events
#     :return: a couple corresponding to the smallest range containing the events
#     """
#     if len(events_gt) == 0:
#         raise ValueError('The gt events should contain at least one event')
#     if len(events_pred) == 0:
#         # empty prediction, base Trange only on events_gt (which is non empty)
#         return(infer_Trange(events_gt, events_gt))
        
#     min_pred = min([x[0] for x in events_pred])
#     min_gt = min([x[0] for x in events_gt])
#     max_pred = max([x[1] for x in events_pred])
#     max_gt = max([x[1] for x in events_gt])
#     Trange = (min(min_pred, min_gt), max(max_pred, max_gt))
#     return(Trange)

# def has_point_anomalies(events):
#     """
#     Checking whether events contain point anomalies, i.e.
#     events starting and stopping at the same time.
    
#     :param events: a list of couples corresponding to predicted events
#     :return: True is the events have any point anomalies, False otherwise
#     """
#     if len(events) == 0:
#         return(False)
#     return(min([x[1] - x[0] for x in events]) == 0)

# def t_start(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
#     """
#     Helper for `E_gt_func`
    
#     :param j: index from 0 to len(Js) (included) on which to get the start
#     :param Js: ground truth events, as a list of couples
#     :param Trange: range of the series where Js is included
#     :return: generalized start such that the middle of t_start and t_stop 
#     always gives the affiliation zone
#     """
#     b = max(Trange)
#     n = len(Js)
#     if j == n:
#         return(2*b - t_stop(n-1, Js, Trange))
#     else:
#         return(Js[j][0])

# def t_stop(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
#     """
#     Helper for `E_gt_func`
    
#     :param j: index from 0 to len(Js) (included) on which to get the stop
#     :param Js: ground truth events, as a list of couples
#     :param Trange: range of the series where Js is included
#     :return: generalized stop such that the middle of t_start and t_stop 
#     always gives the affiliation zone
#     """
#     if j == -1:
#         a = min(Trange)
#         return(2*a - t_start(0, Js, Trange))
#     else:
#         return(Js[j][1])

# def E_gt_func(j, Js, Trange):
#     """
#     Get the affiliation zone of element j of the ground truth
    
#     :param j: index from 0 to len(Js) (excluded) on which to get the zone
#     :param Js: ground truth events, as a list of couples
#     :param Trange: range of the series where Js is included, can 
#     be (-math.inf, math.inf) for distance measures
#     :return: affiliation zone of element j of the ground truth represented
#     as a couple
#     """
#     range_left = (t_stop(j-1, Js, Trange) + t_start(j, Js, Trange))/2
#     range_right = (t_stop(j, Js, Trange) + t_start(j+1, Js, Trange))/2
#     return((range_left, range_right))

# def get_all_E_gt_func(Js, Trange):
#     """
#     Get the affiliation partition from the ground truth point of view
    
#     :param Js: ground truth events, as a list of couples
#     :param Trange: range of the series where Js is included, can 
#     be (-math.inf, math.inf) for distance measures
#     :return: affiliation partition of the events
#     """
#     # E_gt is the limit of affiliation/attraction for each ground truth event
#     E_gt = [E_gt_func(j, Js, Trange) for j in range(len(Js))]
#     return(E_gt)

# def get_pivot_j(I, J):
#     """
#     Get the single point of J that is the closest to I, called 'pivot' here,
#     with the requirement that I should be outside J
    
#     :param I: a non empty interval (start, stop)
#     :param J: another non empty interval, with empty intersection with I
#     :return: the element j of J that is the closest to I
#     """
#     if interval_intersection(I, J) is not None:
#         raise ValueError('I and J should have a void intersection')

#     j_pivot = None # j_pivot is a border of J
#     if max(I) <= min(J):
#         j_pivot = min(J)
#     elif min(I) >= max(J):
#         j_pivot = max(J)
#     else:
#         raise ValueError('I should be outside J')
#     return(j_pivot)

# def integral_mini_interval(I, J):
#     """
#     In the specific case where interval I is located outside J,
#     integral of distance from x to J over the interval x \in I.
#     This is the *integral* i.e. the sum.
#     It's not the mean (not divided by the length of I yet)
    
#     :param I: a interval (start, stop), or None
#     :param J: a non empty interval, with empty intersection with I
#     :return: the integral of distances d(x, J) over x \in I
#     """
#     if I is None:
#         return(0)

#     j_pivot = get_pivot_j(I, J)
#     a = min(I)
#     b = max(I)
#     return((b-a)*abs((j_pivot - (a+b)/2)))

# def cut_into_three_func(I, J):
#     """
#     Cut an interval I into a partition of 3 subsets:
#         the elements before J,
#         the elements belonging to J,
#         and the elements after J
    
#     :param I: an interval represented by start and stop, or None for an empty one
#     :param J: a non empty interval
#     :return: a triplet of three intervals, each represented by either (start, stop) or None
#     """
#     if I is None:
#         return((None, None, None))
    
#     I_inter_J = interval_intersection(I, J)
#     if I == I_inter_J:
#         I_before = None
#         I_after = None
#     elif I[1] <= J[0]:
#         I_before = I
#         I_after = None
#     elif I[0] >= J[1]:
#         I_before = None
#         I_after = I
#     elif (I[0] <= J[0]) and (I[1] >= J[1]):
#         I_before = (I[0], I_inter_J[0])
#         I_after = (I_inter_J[1], I[1])
#     elif I[0] <= J[0]:
#         I_before = (I[0], I_inter_J[0])
#         I_after = None
#     elif I[1] >= J[1]:
#         I_before = None
#         I_after = (I_inter_J[1], I[1])
#     else:
#         raise ValueError('unexpected unconsidered case')
#     return(I_before, I_inter_J, I_after)


# def integral_interval_distance(I, J):
#     """
#     For any non empty intervals I, J, compute the
#     integral of distance from x to J over the interval x \in I.
#     This is the *integral* i.e. the sum. 
#     It's not the mean (not divided by the length of I yet)
#     The interval I can intersect J or not
    
#     :param I: a interval (start, stop), or None
#     :param J: a non empty interval
#     :return: the integral of distances d(x, J) over x \in I
#     """
#     # I and J are single intervals (not generic sets)
#     # I is a predicted interval in the range of affiliation of J
    
#     def f(I_cut):
#         return(integral_mini_interval(I_cut, J))
#     # If I_middle is fully included into J, it is
#     # the distance to J is always 0
#     def f0(I_middle):
#         return(0)

#     cut_into_three = cut_into_three_func(I, J)
#     # Distance for now, not the mean:
#     # Distance left: Between cut_into_three[0] and the point min(J)
#     d_left = f(cut_into_three[0])
#     # Distance middle: Between cut_into_three[1] = I inter J, and J
#     d_middle = f0(cut_into_three[1])
#     # Distance right: Between cut_into_three[2] and the point max(J)
#     d_right = f(cut_into_three[2])
#     # It's an integral so summable
#     return(d_left + d_middle + d_right)

# def affiliation_precision_distance(Is = [(1,2),(3,4),(5,6)], J = (2,5.5)):
#     """
#     Compute the individual average distance from Is to a single ground truth J

#     :param Is: list of predicted events within the affiliation zone of J
#     :param J: couple representating the start and stop of a ground truth interval
#     :return: individual average precision directed distance number
#     """
#     if all([I is None for I in Is]): # no prediction in the current area
#         return(math.nan) # undefined
#     return(sum([integral_interval_distance(I, J) for I in Is]) / sum_interval_lengths(Is))

# def sum_interval_lengths(Is = [(1,2),(3,4),(5,6)]):
#     """
#     Sum of length of the intervals
    
#     :param Is: list of intervals represented by starts and stops
#     :return: sum of the interval length
#     """
#     return(sum([interval_length(I) for I in Is]))

# def interval_length(J = (1,2)):
#     """
#     Length of an interval
    
#     :param J: couple representating the start and stop of an interval, or None
#     :return: length of the interval, and 0 for a None interval
#     """
#     if J is None:
#         return(0)
#     return(J[1] - J[0])

# def integral_interval_probaCDF_precision(I, J, E):
#     """
#     Integral of the probability of distances over the interval I.
#     Compute the integral $\int_{x \in I} Fbar(dist(x,J)) dx$.
#     This is the *integral* i.e. the sum (not the mean)
    
#     :param I: a single (non empty) predicted interval in the zone of affiliation of J
#     :param J: ground truth interval
#     :param E: affiliation/influence zone for J
#     :return: the integral $\int_{x \in I} Fbar(dist(x,J)) dx$
#     """
#     # I and J are single intervals (not generic sets)
#     def f(I_cut):
#         if I_cut is None:
#             return(0)
#         else:
#             return(integral_mini_interval_Pprecision_CDFmethod(I_cut, J, E))
            
#     # If I_middle is fully included into J, it is
#     # integral of 1 on the interval I_middle, so it's |I_middle|
#     def f0(I_middle):
#         if I_middle is None:
#             return(0)
#         else:
#             return(max(I_middle) - min(I_middle))
    
#     cut_into_three = cut_into_three_func(I, J)
#     # Distance for now, not the mean:
#     # Distance left: Between cut_into_three[0] and the point min(J)
#     d_left = f(cut_into_three[0])
#     # Distance middle: Between cut_into_three[1] = I inter J, and J
#     d_middle = f0(cut_into_three[1])
#     # Distance right: Between cut_into_three[2] and the point max(J)
#     d_right = f(cut_into_three[2])
#     # It's an integral so summable
#     return(d_left + d_middle + d_right)

# def affiliation_precision_proba(Is = [(1,2),(3,4),(5,6)], J = (2,5.5), E = (0,8)):
#     """
#     Compute the individual precision probability from Is to a single ground truth J

#     :param Is: list of predicted events within the affiliation zone of J
#     :param J: couple representating the start and stop of a ground truth interval
#     :param E: couple representing the start and stop of the zone of affiliation of J
#     :return: individual precision probability in [0, 1], or math.nan if undefined
#     """
#     if all([I is None for I in Is]): # no prediction in the current area
#         return(0.5) # undefined
#     return(sum([integral_interval_probaCDF_precision(I, J, E) for I in Is]) / sum_interval_lengths(Is))


# def cut_J_based_on_mean_func(J, e_mean):
#     """
#     Helper function for the recall.
#     Partition J into two intervals: before and after e_mean
#     (e_mean represents the center element of E the zone of affiliation)
    
#     :param J: ground truth interval
#     :param e_mean: a float number (center value of E)
#     :return: a couple partitionning J into (J_before, J_after)
#     """
#     if J is None:
#         J_before = None
#         J_after = None
#     elif e_mean >= max(J):
#         J_before = J
#         J_after = None
#     elif e_mean <= min(J):
#         J_before = None
#         J_after = J
#     else: # e_mean is across J
#         J_before = (min(J), e_mean)
#         J_after = (e_mean, max(J))
        
#     return((J_before, J_after))


# def integral_mini_interval_Precall_CDFmethod(I, J, E):
#     """
#     Integral of the probability of distances over the interval J.
#     In the specific case where interval J is located outside I,
#     compute the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$.
#     This is the *integral* i.e. the sum (not the mean)
    
#     :param I: a single (non empty) predicted interval
#     :param J: ground truth (non empty) interval, with empty intersection with I
#     :param E: the affiliation/influence zone for J, represented as a couple (start, stop)
#     :return: the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$
#     """
#     # The interval J should be located outside I 
#     # (so it's either the left piece or the right piece w.r.t I)
#     i_pivot = get_pivot_j(J, I)
#     e_min = min(E)
#     e_max = max(E)
#     e_mean = (e_min + e_max) / 2
    
#     # If i_pivot is outside E (it's possible), then
#     # the distance is worst that any random element within E,
#     # so we set the recall to 0
#     if i_pivot <= min(E):
#         return(0)
#     elif i_pivot >= max(E):
#         return(0)
#     # Otherwise, we have at least i_pivot in E and so d < M so min(d,M)=d
    
#     cut_J_based_on_e_mean = cut_J_based_on_mean_func(J, e_mean)
#     J_before = cut_J_based_on_e_mean[0]
#     J_after = cut_J_based_on_e_mean[1]
  
#     iemin_mean = (e_min + i_pivot)/2
#     cut_Jbefore_based_on_iemin_mean = cut_J_based_on_mean_func(J_before, iemin_mean)
#     J_before_closeE = cut_Jbefore_based_on_iemin_mean[0] # before e_mean and closer to e_min than i_pivot ~ J_before_before
#     J_before_closeI = cut_Jbefore_based_on_iemin_mean[1] # before e_mean and closer to i_pivot than e_min ~ J_before_after
    
#     iemax_mean = (e_max + i_pivot)/2
#     cut_Jafter_based_on_iemax_mean = cut_J_based_on_mean_func(J_after, iemax_mean)
#     J_after_closeI = cut_Jafter_based_on_iemax_mean[0] # after e_mean and closer to i_pivot than e_max ~ J_after_before
#     J_after_closeE = cut_Jafter_based_on_iemax_mean[1] # after e_mean and closer to e_max than i_pivot ~ J_after_after
    
#     if J_before_closeE is not None:
#         j_before_before_min = min(J_before_closeE) # == min(J)
#         j_before_before_max = max(J_before_closeE)
#     else:
#         j_before_before_min = math.nan
#         j_before_before_max = math.nan
  
#     if J_before_closeI is not None:
#         j_before_after_min = min(J_before_closeI) # == j_before_before_max if existing
#         j_before_after_max = max(J_before_closeI) # == max(J_before)
#     else:
#         j_before_after_min = math.nan
#         j_before_after_max = math.nan
   
#     if J_after_closeI is not None:
#         j_after_before_min = min(J_after_closeI) # == min(J_after)
#         j_after_before_max = max(J_after_closeI) 
#     else:
#         j_after_before_min = math.nan
#         j_after_before_max = math.nan
    
#     if J_after_closeE is not None:
#         j_after_after_min = min(J_after_closeE) # == j_after_before_max if existing
#         j_after_after_max = max(J_after_closeE) # == max(J)
#     else:
#         j_after_after_min = math.nan
#         j_after_after_max = math.nan
  
#     # <-- J_before_closeE --> <-- J_before_closeI --> <-- J_after_closeI --> <-- J_after_closeE -->
#     # j_bb_min       j_bb_max j_ba_min       j_ba_max j_ab_min      j_ab_max j_aa_min      j_aa_max
#     # (with `b` for before and `a` for after in the previous variable names)
    
#     #                                          vs e_mean  m = min(t-e_min, e_max-t)  d=|i_pivot-t|   min(d,m)                            \int min(d,m)dt   \int d dt        \int_(min(d,m)+d)dt                                    \int_{t \in J}(min(d,m)+d)dt
#     # Case J_before_closeE & i_pivot after J   before     t-e_min                    i_pivot-t       min(i_pivot-t,t-e_min) = t-e_min    t^2/2-e_min*t     i_pivot*t-t^2/2  t^2/2-e_min*t+i_pivot*t-t^2/2 = (i_pivot-e_min)*t      (i_pivot-e_min)*tB - (i_pivot-e_min)*tA = (i_pivot-e_min)*(tB-tA)
#     # Case J_before_closeI & i_pivot after J   before     t-e_min                    i_pivot-t       min(i_pivot-t,t-e_min) = i_pivot-t  i_pivot*t-t^2/2   i_pivot*t-t^2/2  i_pivot*t-t^2/2+i_pivot*t-t^2/2 = 2*i_pivot*t-t^2      2*i_pivot*tB-tB^2 - 2*i_pivot*tA + tA^2 = 2*i_pivot*(tB-tA) - (tB^2 - tA^2)
#     # Case J_after_closeI & i_pivot after J    after      e_max-t                    i_pivot-t       min(i_pivot-t,e_max-t) = i_pivot-t  i_pivot*t-t^2/2   i_pivot*t-t^2/2  i_pivot*t-t^2/2+i_pivot*t-t^2/2 = 2*i_pivot*t-t^2      2*i_pivot*tB-tB^2 - 2*i_pivot*tA + tA^2 = 2*i_pivot*(tB-tA) - (tB^2 - tA^2)
#     # Case J_after_closeE & i_pivot after J    after      e_max-t                    i_pivot-t       min(i_pivot-t,e_max-t) = e_max-t    e_max*t-t^2/2     i_pivot*t-t^2/2  e_max*t-t^2/2+i_pivot*t-t^2/2 = (e_max+i_pivot)*t-t^2  (e_max+i_pivot)*tB-tB^2 - (e_max+i_pivot)*tA + tA^2 = (e_max+i_pivot)*(tB-tA) - (tB^2 - tA^2)
#     #
#     # Case J_before_closeE & i_pivot before J  before     t-e_min                    t-i_pivot       min(t-i_pivot,t-e_min) = t-e_min    t^2/2-e_min*t     t^2/2-i_pivot*t  t^2/2-e_min*t+t^2/2-i_pivot*t = t^2-(e_min+i_pivot)*t  tB^2-(e_min+i_pivot)*tB - tA^2 + (e_min+i_pivot)*tA = (tB^2 - tA^2) - (e_min+i_pivot)*(tB-tA)
#     # Case J_before_closeI & i_pivot before J  before     t-e_min                    t-i_pivot       min(t-i_pivot,t-e_min) = t-i_pivot  t^2/2-i_pivot*t   t^2/2-i_pivot*t  t^2/2-i_pivot*t+t^2/2-i_pivot*t = t^2-2*i_pivot*t      tB^2-2*i_pivot*tB - tA^2 + 2*i_pivot*tA = (tB^2 - tA^2) - 2*i_pivot*(tB-tA)
#     # Case J_after_closeI & i_pivot before J   after      e_max-t                    t-i_pivot       min(t-i_pivot,e_max-t) = t-i_pivot  t^2/2-i_pivot*t   t^2/2-i_pivot*t  t^2/2-i_pivot*t+t^2/2-i_pivot*t = t^2-2*i_pivot*t      tB^2-2*i_pivot*tB - tA^2 + 2*i_pivot*tA = (tB^2 - tA^2) - 2*i_pivot*(tB-tA)
#     # Case J_after_closeE & i_pivot before J   after      e_max-t                    t-i_pivot       min(t-i_pivot,e_max-t) = e_max-t    e_max*t-t^2/2     t^2/2-i_pivot*t  e_max*t-t^2/2+t^2/2-i_pivot*t = (e_max-i_pivot)*t      (e_max-i_pivot)*tB - (e_max-i_pivot)*tA = (e_max-i_pivot)*(tB-tA)
    
#     if i_pivot >= max(J):
#         part1_before_closeE = (i_pivot-e_min)*(j_before_before_max - j_before_before_min) # (i_pivot-e_min)*(tB-tA) # j_before_before_max - j_before_before_min
#         part2_before_closeI = 2*i_pivot*(j_before_after_max-j_before_after_min) - (j_before_after_max**2 - j_before_after_min**2) # 2*i_pivot*(tB-tA) - (tB^2 - tA^2) # j_before_after_max - j_before_after_min
#         part3_after_closeI = 2*i_pivot*(j_after_before_max-j_after_before_min) - (j_after_before_max**2 - j_after_before_min**2) # 2*i_pivot*(tB-tA) - (tB^2 - tA^2) # j_after_before_max - j_after_before_min  
#         part4_after_closeE = (e_max+i_pivot)*(j_after_after_max-j_after_after_min) - (j_after_after_max**2 - j_after_after_min**2) # (e_max+i_pivot)*(tB-tA) - (tB^2 - tA^2) # j_after_after_max - j_after_after_min
#         out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
#     elif i_pivot <= min(J):
#         part1_before_closeE = (j_before_before_max**2 - j_before_before_min**2) - (e_min+i_pivot)*(j_before_before_max-j_before_before_min) # (tB^2 - tA^2) - (e_min+i_pivot)*(tB-tA) # j_before_before_max - j_before_before_min
#         part2_before_closeI = (j_before_after_max**2 - j_before_after_min**2) - 2*i_pivot*(j_before_after_max-j_before_after_min) # (tB^2 - tA^2) - 2*i_pivot*(tB-tA) # j_before_after_max - j_before_after_min
#         part3_after_closeI = (j_after_before_max**2 - j_after_before_min**2) - 2*i_pivot*(j_after_before_max - j_after_before_min) # (tB^2 - tA^2) - 2*i_pivot*(tB-tA) # j_after_before_max - j_after_before_min
#         part4_after_closeE = (e_max-i_pivot)*(j_after_after_max - j_after_after_min) # (e_max-i_pivot)*(tB-tA) # j_after_after_max - j_after_after_min
#         out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
#     else:
#         raise ValueError('The i_pivot should be outside J')
    
#     out_integral_min_dm_plus_d = _sum_wo_nan(out_parts) # integral on all J, i.e. sum of the disjoint parts

#     # We have for each point t of J:
#     # \bar{F}_{t, recall}(d) = 1 - (1/|E|) * (min(d,m) + d)
#     # Since t is a single-point here, and we are in the case where i_pivot is inside E.
#     # The integral is then given by:
#     # C = \int_{t \in J} \bar{F}_{t, recall}(D(t)) dt
#     #   = \int_{t \in J} 1 - (1/|E|) * (min(d,m) + d) dt
#     #   = |J| - (1/|E|) * [\int_{t \in J} (min(d,m) + d) dt]
#     #   = |J| - (1/|E|) * out_integral_min_dm_plus_d    
#     DeltaJ = max(J) - min(J)
#     DeltaE = max(E) - min(E)
#     C = DeltaJ - (1/DeltaE) * out_integral_min_dm_plus_d
    
#     return(C)


# def integral_interval_probaCDF_recall(I, J, E):
#     """
#     Integral of the probability of distances over the interval J.
#     Compute the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$.
#     This is the *integral* i.e. the sum (not the mean)

#     :param I: a single (non empty) predicted interval
#     :param J: ground truth (non empty) interval
#     :param E: the affiliation/influence zone for J
#     :return: the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$
#     """
#     # I and J are single intervals (not generic sets)
#     # E is the outside affiliation interval of J (even for recall!)
#     # (in particular J \subset E)
#     #
#     # J is the portion of the ground truth affiliated to I
#     # I is a predicted interval (can be outside E possibly since it's recall)
#     def f(J_cut):
#         if J_cut is None:
#             return(0)
#         else:
#             return integral_mini_interval_Precall_CDFmethod(I, J_cut, E)

#     # If J_middle is fully included into I, it is
#     # integral of 1 on the interval J_middle, so it's |J_middle|
#     def f0(J_middle):
#         if J_middle is None:
#             return(0)
#         else:
#             return(max(J_middle) - min(J_middle))
    
#     cut_into_three = cut_into_three_func(J, I) # it's J that we cut into 3, depending on the position w.r.t I
#     # since we integrate over J this time.
#     #
#     # Distance for now, not the mean:
#     # Distance left: Between cut_into_three[0] and the point min(I)
#     d_left = f(cut_into_three[0])
#     # Distance middle: Between cut_into_three[1] = J inter I, and I
#     d_middle = f0(cut_into_three[1])
#     # Distance right: Between cut_into_three[2] and the point max(I)
#     d_right = f(cut_into_three[2])
#     # It's an integral so summable
#     return(d_left + d_middle + d_right)

# def affiliation_recall_proba(Is = [(1,2),(3,4),(5,6)], J = (2,5.5), E = (0,8)):
#     """
#     Compute the individual recall probability from a single ground truth J to Is

#     :param Is: list of predicted events within the affiliation zone of J
#     :param J: couple representating the start and stop of a ground truth interval
#     :param E: couple representing the start and stop of the zone of affiliation of J
#     :return: individual recall probability in [0, 1]
#     """
#     Is = [I for I in Is if I is not None] # filter possible None in Is
#     if len(Is) == 0: # there is no prediction in the current area
#         return(0)
#     E_gt_recall = get_all_E_gt_func(Is, E) # here from the point of view of the predictions
#     Js = affiliation_partition([J], E_gt_recall) # partition of J depending of proximity with Is
#     Js = [J for J in Js if len(J) > 0]
#     return(sum([integral_interval_probaCDF_recall(I, J[0], E) for I, J in zip(Is, Js)]) / interval_length(J))

# def _len_wo_nan(vec):
#     """
#     Count of elements, ignoring math.isnan ones
    
#     :param vec: vector of floating numbers
#     :return: count of the elements, ignoring math.isnan ones
#     """
#     vec_wo_nan = [e for e in vec if not math.isnan(e)]
#     return(len(vec_wo_nan))

# def _sum_wo_nan(vec):
#     """
#     Sum of elements, ignoring math.isnan ones
    
#     :param vec: vector of floating numbers
#     :return: sum of the elements, ignoring math.isnan ones
#     """
#     vec_wo_nan = [e for e in vec if not math.isnan(e)]
#     return(sum(vec_wo_nan))

# def pr_from_events(events_pred, events_gt, Trange):
#     """
#     Compute the affiliation metrics including the precision/recall in [0,1],
#     along with the individual precision/recall distances and probabilities

#     :param events_pred: list of predicted events, each represented by a couple
#     indicating the start and the stop of the event
#     :param events_gt: list of ground truth events, each represented by a couple
#     indicating the start and the stop of the event
#     :param Trange: range of the series where events_pred and events_gt are included,
#     represented as a couple (start, stop)
#     :return: dictionary with precision, recall, and the individual metrics
#     """
#     # testing the inputs
#     test_events(events_pred)
#     test_events(events_gt)

#     # other tests
#     minimal_Trange = infer_Trange(events_pred, events_gt)
#     if not Trange[0] <= minimal_Trange[0]:
#         raise ValueError('`Trange` should include all the events')
#     if not minimal_Trange[1] <= Trange[1]:
#         raise ValueError('`Trange` should include all the events')

#     if len(events_gt) == 0:
#         raise ValueError('Input `events_gt` should have at least one event')

#     if has_point_anomalies(events_pred) or has_point_anomalies(events_gt):
#         raise ValueError('Cannot manage point anomalies currently')

#     if Trange is None:
#         # Set as default, but Trange should be indicated if probabilities are used
#         raise ValueError('Trange should be indicated (or inferred with the `infer_Trange` function')

#     E_gt = get_all_E_gt_func(events_gt, Trange)
#     aff_partition = affiliation_partition(events_pred, E_gt)

#     # # Computing precision distance
#     # d_precision = [affiliation_precision_distance(Is, J) for Is, J in zip(aff_partition, events_gt)]
#     #
#     # # Computing specificity distance
#     # d_specificity = [affiliation_specificity_distance(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]
#     #
#     # # Computing recall distance
#     # d_recall = [affiliation_recall_distance(Is, J) for Is, J in zip(aff_partition, events_gt)]

#     # Computing precision
#     p_precision = [affiliation_precision_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

#     # # Computing specificity
#     # p_specificity = [affiliation_specificity_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

#     # Computing recall
#     p_recall = [affiliation_recall_proba(Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

#     if _len_wo_nan(p_precision) > 0:
#         p_precision_average = _sum_wo_nan(p_precision) / _len_wo_nan(p_precision)
#     else:
#         p_precision_average = p_precision[0] # math.nan
#     # p_specificity_average = sum(p_specificity) / len(p_specificity)
#     p_recall_average = sum(p_recall) / len(p_recall)

#     dict_out = dict({'precision': p_precision_average,
#                      #'specificity': p_specificity_average,
#                      'recall': p_recall_average,
#                      'affiliation_zones': E_gt,
#                      'individual_precision_probabilities': p_precision,
#                      #'individual_specificity_probabilities': p_specificity,
#                      'individual_recall_probabilities': p_recall})
#     return(dict_out)


# print(f"{grouped_anomalies['time_start'][0]=}")
# print(f"{grouped_anomalies['time_end'][0]=}")
# print(f"{true_anomalies['StartTime'][0]=}")
# print(f"{true_anomalies['EndTime'][0]=}")

# # Step 1: Extract events_pred from grouped_anomalies
# events_pred = [
#     (row['time_start'], row['time_end']) 
#     for _, row in grouped_anomalies.iterrows() 
# ]

# # Step 2: Extract events_gt from true_anomalies
# events_gt = [
#     (row['StartTime'], row['EndTime']) 
#     for _, row in true_anomalies.iterrows()
# ]

# # Step 3: Define the time range
# Trange = (
#     min([event[0] for event in events_pred + events_gt]),  # Start of the time range
#     max([event[1] for event in events_pred + events_gt])   # End of the time range
# )

# # Step 4: Call the function
# result = pr_from_events(events_pred, events_gt, Trange)

# # Output the result
# print(result)