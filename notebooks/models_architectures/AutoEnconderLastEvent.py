import math
from datetime import datetime
from libraries.utils import read_csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn

from libraries.sequence_generators import sequence_generator_last_event

# class SinusoidalPositionalEncoding(nn.Module):
#     def __init__(self, sequence_length, n_features):
#         super(SinusoidalPositionalEncoding, self).__init__()
        
#         # Creamos las posiciones y los índices
#         position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)  # [sequence_length, 1]
#         div_term = torch.exp(torch.arange(0, n_features, 2).float() * (-math.log(10000.0) / n_features))  # [n_features/2]

#         # Calculamos senos y cosenos en paralelo
#         pe = torch.zeros(sequence_length, n_features)
#         pe[:, 0::2] = torch.sin(position * div_term)  # Dimensiones pares
#         pe[:, 1::2] = torch.cos(position * div_term)  # Dimensiones impares

#         # Registramos el buffer como no entrenable
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe.unsqueeze(0)  # Agregamos positional encoding
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, sequence_length, n_features):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        # Creamos las posiciones y los índices
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)  # [sequence_length, 1]
        div_term = torch.exp(torch.arange(0, n_features, 2).float() * (-math.log(10000.0) / n_features))  # [n_features//2]

        # Calculamos senos y cosenos en paralelo
        pe = torch.zeros(sequence_length, n_features)
        pe[:, 0::2] = torch.sin(position * div_term)  # Dimensiones pares
        if n_features % 2 == 1:  # Si n_features es impar
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Ajustamos el tamaño de div_term
        else:
            pe[:, 1::2] = torch.cos(position * div_term)  # Dimensiones impares

        # Registramos el buffer como no entrenable
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe.unsqueeze(0)  # Agregamos positional encoding
    
class AutoEnconderLastEvent(nn.Module):
    def __init__(self, sequence_length, n_features, latent_dim):
        super(AutoEnconderLastEvent, self).__init__()

        # Positional Encoding Sinusoidal
        self.positional_encoding = SinusoidalPositionalEncoding(sequence_length, n_features)

        ## ENCODER
        self.attention = nn.MultiheadAttention(embed_dim=n_features, num_heads=1, batch_first=True)
        self.layer_norm = nn.LayerNorm((sequence_length, n_features))
        self.encoder_fc_1 = nn.Linear(n_features, 4 * latent_dim)
        self.encoder_fc_2 = nn.Linear(4 * latent_dim, 2 * latent_dim)
        self.encoder_fc_3 = nn.Linear(2 * latent_dim, latent_dim)
        self.relu = nn.ReLU()

        # LATENT REPRESENTATION
        self.attention_summary = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=2, batch_first=True)

        ## DECODER
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, 4 * latent_dim),
            nn.ReLU(),
            nn.Linear(4 * latent_dim, n_features),  # Salida ajustada para reconstruir solo el último elemento
        )

    def forward(self, x):
        # Aplicar atención multi-cabezal
        attn_output, _ = self.attention(x, x, x)
        # Normalización
        attn_output = self.layer_norm(attn_output)
        # Capas densas y activación
        encoded_1 = self.relu(self.encoder_fc_1(attn_output))
        encoded_2 = self.relu(self.encoder_fc_2(encoded_1))
        encoded_3 = self.relu(self.encoder_fc_3(encoded_2))  # Representación latente de toda la secuencia

        # Opción 1: Promediar las representaciones a través de la dimensión temporal para obtener un resumen global
        # latent_vector = encoded_3.mean(dim=1)  # Dimensión [batch_size, latent_dim]

        # Opción 2: Aplicar atención multi-cabezal a la representación latente
        summary, _ = self.attention_summary(encoded_3, encoded_3, encoded_3)
        latent_vector = summary.mean(dim=1)  

        # Decodificación desde la representación latente
        decoded = self.decoder(latent_vector)  # Dimensión [batch_size, n_features]
        return decoded
    
    def predict(self, threshold_list, df, window_size, batch_size, device):
        generator = sequence_generator_last_event(df.values, window_size, batch_size)
        steps = (len(df) - window_size) // batch_size

        # Generar las secuencias
        anomalies = []
        for _ in tqdm(range(steps), desc="Detectando anomalías"):
            batch_inputs, batch_targets = next(generator)  # Obtener inputs y targets
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Reconstruir últimas filas
            reconstructed_batch = self.forward(batch_inputs).detach().cpu().numpy()
            batch_targets = batch_targets.cpu().numpy()  # Convertir targets a numpy

            # Calcular el error de reconstrucción (últimas filas)
            reconstruction_error = np.square(batch_targets - reconstructed_batch)

            # Detectar anomalías (1 si el error excede el umbral)
            anomalies.extend((reconstruction_error > threshold_list).astype(int).tolist())

        # Convertir anomalías en un DataFrame
        first_el = window_size - 1
        last_el = first_el + len(anomalies)
        return pd.DataFrame(anomalies, columns=df.columns, index=df.index[first_el:last_el])
