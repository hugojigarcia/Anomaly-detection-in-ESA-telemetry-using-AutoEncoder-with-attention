import pandas as pd
import numpy as np
from pitia.backend.backend import Backend
from datetime import datetime
from pitia.models.model import Model
import pandas as pd
import pickle
from libraries.utils import read_csv



def import_data(input_data_path, channels):
    def __check_import_data_format(result):
        # Comprobamos que el dataset tiene la salida identica a lo que necesita pitia
        if not isinstance(result.index, pd.DatetimeIndex):
            raise ValueError("El índice del DataFrame no es un DateTimeIndex.")

        if result.isnull().any().any():
            raise ValueError("Hay valores NaN en el DataFrame.")
            
        if result.index.name != 'time':
            raise ValueError("El índice debe llamarse time")

    result = read_csv(input_data_path, sep=";")
    result.index = pd.to_datetime(result.index)
    if channels is not None:
        result = result[channels]
    __check_import_data_format(result)
    return result


import polars as pl

def import_data_v2(input_data_path, start_date = None, end_date = None,channels=None):
    result = pl.scan_csv(input_data_path, separator=";",n_rows=10)
    
    schema = {}
    schema['time'] = pl.Datetime
    columns =result.collect_schema().names()
    
    for column in columns:
        if "time" not in column:
            schema[column] = pl.Float32

    
    result = pl.scan_csv(input_data_path, separator=";", schema=schema)
    
    if start_date is not None:
        result = result.filter(pl.col('time') >= start_date)
        result = result.filter(pl.col('time') <= end_date)
 
    if channels is not None:
        result = result.select(['time'] + channels)

    result = result.collect()
    result = result.to_pandas()
    result.set_index('time', inplace=True)
    
    return result



def replace_outliers(result, remove_outliers, start_date_train, end_date_train):
    def replace_outliers_process(data):
        total_outliers = 0
        for columna in data.columns:
            if data[columna].dtype in ['float64', 'int64']:  
                Q1 = data[columna].quantile(0.25)
                Q3 = data[columna].quantile(0.75)
                IQR = Q3 - Q1
                filtro = ~((data[columna] >= (Q1 - 1.5 * IQR)) & (data[columna] <= (Q3 + 1.5 * IQR)))
                data[columna][filtro] = np.nan
                total_outliers += len([el for el in filtro if el])
        print(f"Total data: {len(data)}")
        print(f"Total outliers: {total_outliers}")
        return data
    if remove_outliers:
        # datos_entrenamiento = result[result.index.month == 3]
        datos_entrenamiento = result[(result.index >= start_date_train) & (result.index < end_date_train)]
        datos_entrenamiento_sin_outliers = replace_outliers_process(datos_entrenamiento.copy())
        result.update(datos_entrenamiento_sin_outliers)
    return result


def get_model_backend(df):
    backend = Backend()
    backend.df = df
    backend.local_backend = True
    return backend

def get_num_components_needed(backend, df, start_date_train, end_date_train, min_var_explained):
    model = Model(backend=backend,
                n_components=len(df.columns),
                start_date=start_date_train,
                end_date=end_date_train,
                x_cols=list(df.columns),
                dropna=True,
                filter_pocs=False
                )

    explained_variance = model._sklearn_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    return np.argmax(cumulative_variance >= min_var_explained) + 1

def train_model_process(backend, df, n_components, start_date_train, end_date_train):
    return Model(backend=backend,
              n_components=n_components,
              start_date=start_date_train,
              end_date=end_date_train,
              x_cols=list(df.columns),
              dropna=True,
              filter_pocs=False
             )
    
def train_model_process_v2(backend, df, n_components, start_date_train, end_date_train, remove_outliers):
    return Model(backend=backend,
              n_components=n_components,
              start_date=start_date_train,
              end_date=end_date_train,
              x_cols=list(df.columns),
              dropna=True,
              filter_pocs=remove_outliers
             )
    
def save_model(model, save_path):
    if save_path is not None:
        with open(save_path, 'wb') as file:
            pickle.dump(model, file)


def calculate_contribution_columns(model, val_data):
    print("CALCULATING RANKING COLUMNS")
    contributions = np.abs(pd.DataFrame(model.observation_to_SPE_contributions(val_data.values), index=val_data.index, columns=val_data.columns))
    k = len(val_data.columns)

    row_sums = contributions.sum(axis=1)
    percentages = contributions.div(row_sums, axis=0) * 100
    top_k_cols = np.argsort(-contributions.values, axis=1)[:, :k]
    ranking_cols = []
    for i, row in enumerate(top_k_cols):
        cols = [f"{contributions.columns[col]} ({percentages.iat[i, col]:.2f}%)" for col in row]
        ranking_cols.append(cols)
    print("FINISH RANKING COLUMNS")
    return ranking_cols

# def predict(model, data, start_date, end_date, ranking_cols = True):
#     predict_data = data[(start_date <= data.index) & (data.index <= end_date)].copy()
#     spe_error = model.SPE(predict_data.to_numpy())
    

def predict(model, data, start_date, end_date, ranking_cols = True):
    val_data = data[(start_date <= data.index) & (data.index <= end_date)].copy()

    # Check if val_data is empty
    if val_data.empty:
        # Create an empty DataFrame with the required columns
        empty_cols = data.columns.tolist() + ['SPE_error', 'T2_error']
        if ranking_cols:
            empty_cols.append('Ranking_cols')
        return pd.DataFrame(columns=empty_cols)
    
    # SPE error
    spe_error = model.SPE(val_data.to_numpy())
    T2_error = model.T2(val_data.to_numpy())
    val_data.loc[:, 'SPE_error'] = spe_error
    val_data.loc[:,"T2_error"] = T2_error
    # val_data = val_data[val_data['SPE_error'] > model.UCL_SPE]
    # logs = val_data.copy()
    # val_data.drop(columns=['SPE_error'], inplace=True)
    # Ranking cols
    if ranking_cols:
        ranking_cols = calculate_contribution_columns(model, val_data)
        # matching_indices = val_data.index.isin(logs.index)
        # row_numbers = np.where(matching_indices)[0]
        # Prepare logs
        # logs['Ranking_cols'] = [ranking_cols[i] for i in row_numbers]
        # logs.loc[:, 'Ranking_cols'] = ranking_cols
        logs = logs.assign(Ranking_cols=ranking_cols)
    return val_data

def save_logs(logs, id_prueba, ucl, alfa_ucl, min_var_explained, n_components, remove_outliers, save_path):
    logs = logs[['SPE_error', 'Ranking_cols']]
    logs = logs.assign(**{'ID Prueba': id_prueba})
    logs = logs.assign(UCL=ucl)
    logs = logs.assign(ALFA_UCL=alfa_ucl)
    logs = logs.assign(**{'Varianza explicada (%)': min_var_explained * 100})
    logs = logs.assign(**{'Componentes PCA': n_components})
    logs = logs.assign(**{'Outliers entrenamiento': 'REMOVED' if remove_outliers else 'NO REMOVED'})
    if save_path is not None:
        logs.to_csv(save_path)
    return logs

def train_model_by_variance(input_data_path,
                            channels,
                            remove_outliers,
                            start_date_train,
                            end_date_train,
                            min_var_explained,
                            model_save_path):
    data = import_data(input_data_path, channels)
    data = replace_outliers(data, remove_outliers, start_date_train, end_date_train)
    backend = get_model_backend(data)
    n_components_needed = get_num_components_needed(backend, data, start_date_train, end_date_train, min_var_explained)
    model = train_model_process_v2(backend, data, n_components_needed, start_date_train, end_date_train,remove_outliers)
    save_model(model, model_save_path)
    return model

def train_model_by_variance_v2(data,
                            remove_outliers,
                            start_date_train,
                            end_date_train,
                            min_var_explained,
                            model_save_path):
    # data = replace_outliers(data, remove_outliers, start_date_train, end_date_train)
    backend = get_model_backend(data)
    n_components_needed = get_num_components_needed(backend, data, start_date_train, end_date_train, min_var_explained)
    model = train_model_process(backend, data, n_components_needed, start_date_train, end_date_train)
    save_model(model, model_save_path)
    return model

def train_model_by_components(input_data_path,
                            channels,
                            remove_outliers,
                            start_date_train,
                            end_date_train,
                            components,
                            model_save_path):
    data = import_data(input_data_path, channels)
    data = replace_outliers(data, remove_outliers, start_date_train, end_date_train)
    backend = get_model_backend(data)
    model = train_model_process(backend, data, components, start_date_train, end_date_train)
    save_model(model, model_save_path)
    return model
    
    
    
def train_and_validate_model(id_prueba,
                             input_data_path,
                             channels,
                             remove_outliers,
                             start_date_train,
                             end_date_train,
                             start_date_val,
                             end_date_val,
                             min_var_explained,
                             model_save_path,
                             logs_save_path):
    data = import_data(input_data_path, channels)
    data = replace_outliers(data, remove_outliers, start_date_train, end_date_train)
    backend = get_model_backend(data)
    n_components_needed = get_num_components_needed(backend, data, start_date_train, end_date_train, min_var_explained)
    model = train_model_process(backend, data, n_components_needed, start_date_train, end_date_train)
    ucl = model.UCL_SPE
    alfa_ucl = model._calc_UCL_SPE(alpha=0.01)
    save_model(model, model_save_path)
    logs = predict(model, data, start_date_val, end_date_val)
    logs = save_logs(logs, id_prueba, ucl, alfa_ucl, min_var_explained, n_components_needed, remove_outliers, logs_save_path)
    return model, logs


if __name__ == "__main__":
    phase = 1
    input_data_path = f'notebooks/input_data/phase{phase}.csv' ### PARAMETER

    if phase == 1:
        # start_date_train = datetime(2000, 1, 1, 0, 0, 0) ### PARAMETER Phase1
        # end_date_train = datetime(2000, 3, 11, 0, 0, 0, 0) ### PARAMETER Phase1
        # start_date_val = datetime(2000, 3, 11, 0, 0, 0) ### PARAMETER Phase1
        # end_date_val = datetime(2000, 4, 1, 0, 0, 0, 0) ### PARAMETER Phase1
        start_date_train = datetime(2000, 1, 1, 0, 0, 0) ### PARAMETER Phase1
        end_date_train = datetime(2000, 1, 2, 0, 0, 0, 0) ### PARAMETER Phase1
        start_date_val = datetime(2000, 1, 2, 0, 0, 0) ### PARAMETER Phase1
        end_date_val = datetime(2000, 1, 3, 0, 0, 0, 0) ### PARAMETER Phase1
    elif phase == 2:
        start_date_train = datetime(2000, 1, 1, 0, 0, 0) ### PARAMETER Phase2
        end_date_train = datetime(2000, 9, 1, 0, 0, 0, 0) ### PARAMETER Phase2
        start_date_val = datetime(2000, 9, 1, 0, 0, 0) ### PARAMETER Phase2
        end_date_val = datetime(2000, 11, 1, 0, 0, 0, 0) ### PARAMETER Phase2
    elif phase == 3:
        start_date_train = datetime(2000, 1, 1, 0, 0, 0) ### PARAMETER Phase3
        end_date_train = datetime(2001, 7, 1, 0, 0, 0, 0) ### PARAMETER Phase3
        start_date_val = datetime(2001, 7, 1, 0, 0, 0) ### PARAMETER Phase3
        end_date_val = datetime(2001, 11, 1, 0, 0, 0, 0) ### PARAMETER Phase3
    elif phase == 4:
        start_date_train = datetime(2000, 1, 1, 0, 0, 0) ### PARAMETER Phase4
        end_date_train = datetime(2003, 4, 1, 0, 0, 0, 0) ### PARAMETER Phase4
        start_date_val = datetime(2003, 4, 1, 0, 0, 0) ### PARAMETER Phase4
        end_date_val = datetime(2003, 7, 1, 0, 0, 0, 0) ### PARAMETER Phase4
    elif phase == 5:
        start_date_train = datetime(2000, 1, 1, 0, 0, 0) ### PARAMETER Phase5
        end_date_train = datetime(2006, 10, 1, 0, 0, 0, 0) ### PARAMETER Phase5
        start_date_val = datetime(2006, 10, 1, 0, 0, 0) ### PARAMETER Phase5
        end_date_val = datetime(2007, 1, 1, 0, 0, 0, 0) ### PARAMETER Phase5


    channels = None
    # channels = [f"channel_{i}" for i in range(41,47)]

    remove_outliers = False  ### PARAMETER
    min_var_explained = 0.95 ### PARAMETER

    id_prueba = phase ### PARAMETER
    test_type = 'Phase' ### PARAMETER
    # test_type = 'Channels41-46 Phase' ### PARAMETER

    model_save_path = f"notebooks/models/{test_type}{id_prueba}.pkl"
    logs_save_path = f"notebooks/experiments_csv/{test_type}{id_prueba}.csv"

    train_and_validate_model(id_prueba,
                             input_data_path,
                             channels,
                             remove_outliers,
                             start_date_train,
                             end_date_train,
                             start_date_val,
                             end_date_val,
                             min_var_explained,
                             model_save_path,
                             logs_save_path)