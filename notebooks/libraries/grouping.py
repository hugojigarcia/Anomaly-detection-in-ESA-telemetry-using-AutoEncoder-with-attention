import pandas as pd

def group_df(df, X_seconds):
    if len(df) > 0:
        grouped_anomalies = df.copy()

        # Convertir la columna de tiempo a datetime
        grouped_anomalies['time'] = pd.to_datetime(grouped_anomalies['time'])

        # Calcular la diferencia de tiempo entre las filas
        grouped_anomalies['time_diff'] = grouped_anomalies['time'].diff().dt.total_seconds().fillna(0)

        # Identificar los grupos basados en el umbral de X segundos
        grouped_anomalies['group'] = (grouped_anomalies['time_diff'] > X_seconds).cumsum()

        # Agrupar por los grupos y calcular los tiempos de inicio, fin y el número de anomalías
        def aggregate_group(group):
            group_dict = {
                'SPE_error': group['SPE_error_diff'].max(),
                'UCL': group['UCL_diff'].iloc[0],
                'anomalies_count': len(group),
                'time_start': group['time'].iloc[0],
                'time_end': group['time'].iloc[-1],
            }

            # Calcular la influencia total de cada columna
            if "Ranking_cols" in group.columns:
                ranking_cols = {}
                _num_els_group = len(group["Ranking_cols"])
                for el in group["Ranking_cols"]:
                    el_array = el.strip("[]").replace("'", "")
                    el_array = [] if el_array == "" else el_array.split(", ")
                    for channel_info in el_array:
                        _aux = channel_info.split(" ")
                        channel, influnce = _aux[0], float(_aux[1].strip("()%"))
                        if channel not in ranking_cols:
                            ranking_cols[channel] = 0
                        ranking_cols[channel] += influnce
                for channel, value in ranking_cols.items():
                    ranking_cols[channel] = value / _num_els_group
                sorted_channels = sorted(ranking_cols.items(), key=lambda x: x[1], reverse=True)
                group_dict["Ranking_cols"] = [f"{key} ({value:.2f}%)" for key, value in sorted_channels]

            # Añadir las demás columnas manteniendo la primera fila del grupo
            for col in group.columns:
                if col not in group_dict:
                    group_dict[col] = group[col].iloc[0]
            return pd.Series(group_dict)


        grouped_anomalies = grouped_anomalies.groupby('group').apply(aggregate_group).reset_index(drop=True)

        grouped_anomalies = grouped_anomalies.drop(columns=['time_diff', 'group','time']).sort_values(by='time_start')
    else:
        grouped_anomalies = pd.DataFrame(columns=df.columns)
        grouped_anomalies[['anomalies_count', 'time_start', 'time_end']] = None
    return grouped_anomalies