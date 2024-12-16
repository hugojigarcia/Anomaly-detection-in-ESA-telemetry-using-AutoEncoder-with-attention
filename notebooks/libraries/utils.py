import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import json
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm


#%%
def load_mission_config(mission_choise_path="ESA/configs/mission_choise.json",
                        mission1_config_path="ESA/configs/mission_config_mission1.json",
                        mission2_config_path="ESA/configs/mission_config_mission2.json"):
    mission_choise = load_config(mission_choise_path)["mission"]
    if mission_choise == 1:
        return load_config(mission1_config_path)
    elif mission_choise == 2:
        return load_config(mission2_config_path)
    else:
        raise ValueError(f"Invalid mission choice: {mission_choise}. Expected 1 or 2.")


#%%
def load_config(config_path="config.json"):
    with open(config_path, 'r') as file:
        return json.load(file)


#%%
def read_csv(path, sep=';'):
    _df_columns = pd.read_csv(path, index_col=0, sep=sep, nrows=0)
    _dtype_dict = {col: 'float32' for col in _df_columns.columns if col != 'time'}
    # return pd.read_csv(path, index_col=0, sep=sep, dtype=_dtype_dict)
    df = pd.read_csv(path, index_col=0, sep=sep, dtype=_dtype_dict)
    # df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index, format='mixed', errors='coerce').tz_localize(None)
    # df.index = df.index.dt.tz_localize(None)
    return df


#%%
def interpolate_dataframe(data: pd.DataFrame,
                                   start_date: str,
                                   end_date: str,
                                   esa_anomalies: pd.DataFrame,
                                   interpolation_method: str = 'previous',
                                   sample_frequency: str = '18s') -> pd.DataFrame:
    # Obtén los límites de tiempo del DataFrame
    min_date, max_date = get_time_boundaries(data)
    index_range = data.loc[min_date:max_date].index

    # Define una función interna que procesará una columna
    def process_column(column_name):
        # Elimina los valores NaN de la columna antes de la interpolación
        column_data_cleaned = data[[column_name]].dropna()
        
        # Verifica si la columna está vacía después de eliminar los NaN
        if column_data_cleaned.empty:
            raise ValueError(f"Column {column_name} has no data left after removing NaNs and cannot be interpolated.")
        
        # Interpola la columna limpiada y devuelve el resultado
        return interpolate_column(column_data_cleaned,
                                  column_name,
                                  index_range=index_range,
                                  start_date=start_date,
                                  end_date=end_date,
                                  esa_anomalies=esa_anomalies,
                                  interpolation_method=interpolation_method,
                                  sample_frequency=sample_frequency)

    # Utiliza joblib para ejecutar la interpolación en paralelo en cada columna
    interpolated_dataframes = Parallel(n_jobs=-1)(
        delayed(process_column)(column_name) for column_name in tqdm(data.columns, desc="Interpolating columns")
    )

    # Concatenar todos los DataFrames interpolados en un solo DataFrame a lo largo de las columnas
    complete_interpolated_data = pd.concat(interpolated_dataframes, axis=1)

    return complete_interpolated_data


#%%
def interpolate_dataframe_original(data: pd.DataFrame,
                          start_date: str,
                          end_date: str,
                          esa_anomalies: pd.DataFrame,
                          interpolation_method: str = 'previous',
                          sample_frequency: str = '30s') -> pd.DataFrame:
    # List to store the interpolated DataFrames
    interpolated_dataframes = []
    min_date, max_date = get_time_boundaries(data)
    index_range = data.loc[min_date:max_date].index
    # Iterate over each column in the input DataFrame
    for column_name in tqdm(data.columns, desc="Interpolating columns"):
        # Remove NaN values from the current column before interpolation
        column_data_cleaned = data[[column_name]].dropna()

        # Check if the column is completely empty after removing NaNs
        if column_data_cleaned.empty:
            raise ValueError(f"Column {column_name} has no data left after removing NaNs and cannot be interpolated.")
        
        # Interpolate the cleaned data of the column and add the result to the list
        interpolated_column = interpolate_column(column_data_cleaned,
                                                 column_name,
                                                 index_range=index_range,
                                                 start_date=start_date,
                                                 end_date=end_date,
                                                 esa_anomalies=esa_anomalies,
                                                 interpolation_method=interpolation_method,
                                                 sample_frequency=sample_frequency)
        interpolated_dataframes.append(interpolated_column)

    # Concatenate all interpolated DataFrames into a single DataFrame along the columns
    complete_interpolated_data = pd.concat(interpolated_dataframes, axis=1)

    return complete_interpolated_data


#%%
def get_time_boundaries(df: pd.DataFrame) -> tuple:
    min_time = max([df[df.columns[i]].dropna().head(1).index.values[0] for i in range(len(df.columns))])    
    max_time = min([df[df.columns[i]].dropna().tail(1).index.values[0] for i in range(len(df.columns))])    
    min_time, max_time = str(pd.to_datetime (min_time)), str(pd.to_datetime (max_time))
    return min_time, max_time


#%%
def interpolate_column(dataframe: pd.DataFrame,
                       column_name: str,
                       index_range,
                       start_date: str,
                       end_date: str,
                       esa_anomalies: pd.DataFrame,
                       interpolation_method: str = 'previous',
                       sample_frequency: str = '1s'):
    if interpolation_method == 'zero-order':
        return interpolate_column_zero_order_ESA(dataframe,
                                                 column_name,
                                                 start_date,
                                                 end_date,
                                                 esa_anomalies,
                                                 sample_frequency)
    elif interpolation_method == "frequency-previous":
        return interpolate_column_frequency_previous(dataframe,
                                                     column_name,
                                                     start_date,
                                                     end_date,
                                                     sample_frequency)
    else:
        return interpolate_column_without_frecuency(dataframe, column_name, index_range, interpolation_method)


#%%
def interpolate_column_without_frecuency(dataframe: pd.DataFrame,
                                         column_name: str,
                                         index_range, 
                                         interpolation_method: str = 'previous') -> pd.DataFrame:

    # Interpolation process
    interpolator = interp1d(dataframe.index.values.astype(np.int64), dataframe[column_name].values, kind=interpolation_method)
    interpolated_values = interpolator(index_range.values.astype(np.int64))
    
    # Create a new DataFrame with interpolated values
    interpolated_dataframe = pd.DataFrame({column_name: interpolated_values}, index=index_range)
    
    # Name the index 'Time'
    interpolated_dataframe.index.name = 'time'
    
    return interpolated_dataframe


#%%
def interpolate_column_zero_order_ESA(dataframe: pd.DataFrame,
                                      column_name: str,
                                      start_date: str,
                                      end_date: str, 
                                      esa_anomalies: pd.DataFrame,
                                      sample_frequency: str = '30s') -> pd.DataFrame:
    # STEP 1 AND STEP 2
    def _step1_and_step2():
        uniform_timestamps = pd.date_range(start=start_date, end=end_date, freq=sample_frequency)
        channel_values = []
        i = 0
        for row_time, row in dataframe[(dataframe.index >= start_date) & (dataframe.index <= end_date)].iterrows():
            row_value = row[column_name]
            if not pd.isna(row_value):
                if i == 0:
                    channel_values.append(row_value)
                    last_value = row_value
                    i+=1
                while i<len(uniform_timestamps) and row_time >= uniform_timestamps[i]:
                    channel_values.append(last_value)
                    i+=1

                last_value = row_value
        while i < len(uniform_timestamps):
            channel_values.append(last_value)
            i+=1
        return pd.DataFrame({'time': uniform_timestamps, column_name: channel_values})
    
    
    # STEP 3
    def _step3(uniform_df):
        for i, row in uniform_df.iloc[:-1].iterrows():
            row_start_time, row_end_time = row["time"], uniform_df.iloc[i+1]["time"]
            # print(row_start_time, "/", row_end_time)
            esa_anomalies_in_time_range = esa_anomalies[(esa_anomalies["Channel"] == column_name) & 
                ~((esa_anomalies["EndTime"] <= row_start_time) | (esa_anomalies["StartTime"] >= row_end_time))]
            if not esa_anomalies_in_time_range.empty:
                data_in_time_range = dataframe[(dataframe.index >= row_start_time) & (dataframe.index <=row_end_time)]
                anomaly_value_to_change = None
                for data_time, data in data_in_time_range.iloc[::-1].iterrows():
                    data_value = data[column_name]
                    # print("DATA", data_time, data_value)
                    for _, anomaly in esa_anomalies_in_time_range.iterrows():
                        if data_time >= anomaly["StartTime"] and data_time <= anomaly["EndTime"]:
                            # print("AMOMALY:", anomaly["StartTime"], "/", anomaly["EndTime"])
                            anomaly_value_to_change = data_value
                            break
                    if anomaly_value_to_change is not None:
                        break
                if anomaly_value_to_change is not None:
                    uniform_df.loc[i+1, column_name] = anomaly_value_to_change
                
                # print("VALUE TO CHANGE:", anomaly_value_to_change)
            # print()
        return uniform_df

    uniform_df = _step1_and_step2()
    uniform_df = _step3(uniform_df)
    uniform_df.set_index('time', inplace=True)
    return uniform_df


#%%
def interpolate_column_frequency_previous(dataframe: pd.DataFrame,
                                          column_name: str,
                                          start_date: str,
                                          end_date: str, 
                                          sample_frequency: str = '30s') -> pd.DataFrame:
    # Crear un rango de tiempo con la frecuencia deseada
    time_index = pd.date_range(start=start_date, end=end_date, freq=sample_frequency)
    
    # # Asegurar que el índice del DataFrame sea de tipo datetime
    # dataframe_copy = dataframe[(dataframe.index >= start_date) & (dataframe.index <= end_date)].copy()
    # dataframe_copy.index = pd.to_datetime(dataframe_copy.index)
    
    # Crear un DataFrame con el rango de tiempo deseado
    new_dataframe = pd.DataFrame(index=time_index)
    
    # Agregar la columna a interpolar al nuevo DataFrame
    new_dataframe[column_name] = dataframe[column_name]
    
    # Usar forward-fill para interpolar los valores faltantes
    new_dataframe[column_name] = new_dataframe[column_name].fillna(method='ffill')

    # Usar backward-fill para llenar valores iniciales faltantes
    new_dataframe[column_name] = new_dataframe[column_name].fillna(method='bfill')

    # Rename the index to 'time'
    new_dataframe.index.name = 'time'

    return new_dataframe