import torch
import numpy as np

def sequence_generator(data, window_size, batch_size):
    """
    Generador de secuencias para datos grandes.

    Args:
        data: DataFrame o array con los datos.
        window_size: Tamaño de la ventana para las secuencias.
        batch_size: Número de secuencias por lote.

    Yields:
        Un lote de tuplas (inputs, targets) de tamaño (batch_size, window_size, n_features).
    """
    n_samples = len(data)
    while True:
        for start in range(0, n_samples - window_size, batch_size):
            end = min(start + batch_size, n_samples - window_size)
            batch = []
            for i in range(start, end):
                batch.append(data[i : i + window_size])
            batch = torch.tensor(np.array(batch), dtype=torch.float32)
            yield batch, batch  # En autoencoders, inputs y targets son iguales.


def sequence_generator_last_event(data, window_size, batch_size):
    """
    Generador de secuencias para datos grandes.

    Args:
        data: DataFrame o array con los datos.
        window_size: Tamaño de la ventana para las secuencias.
        batch_size: Número de secuencias por lote.

    Yields:
        Un lote de tuplas (inputs, targets), donde:
        - inputs: Tensor de tamaño (batch_size, window_size, n_features).
        - targets: Tensor de tamaño (batch_size, n_features), correspondiendo a la última fila de cada ventana.
    """
    n_samples = len(data)
    while True:
        for start in range(0, n_samples - window_size, batch_size):
            end = min(start + batch_size, n_samples - window_size)
            batch_inputs = []
            batch_targets = []
            for i in range(start, end):
                window = data[i : i + window_size]  # Extraer la ventana
                batch_inputs.append(window)  # Ventana completa como entrada
                batch_targets.append(window[-1])  # Última fila como objetivo
            batch_inputs = torch.tensor(np.array(batch_inputs), dtype=torch.float32)
            batch_targets = torch.tensor(np.array(batch_targets), dtype=torch.float32)
            yield batch_inputs, batch_targets
