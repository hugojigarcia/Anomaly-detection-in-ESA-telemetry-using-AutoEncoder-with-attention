import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import math

from libraries.sequence_generators import sequence_generator


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


class AutoEnconderFullWindow(nn.Module):
    def __init__(self, sequence_length, n_features, latent_dim):
        super(AutoEnconderFullWindow, self).__init__()

        # Positional Encoding Sinusoidal
        self.positional_encoding = SinusoidalPositionalEncoding(sequence_length, n_features)

        ## ENCODER
        self.attention = nn.MultiheadAttention(embed_dim=n_features, num_heads=1, batch_first=True)
        self.layer_norm = nn.LayerNorm((sequence_length, n_features))
        self.encoder_fc_1 = nn.Linear(n_features, 4*latent_dim)
        self.encoder_fc_2 = nn.Linear(4*latent_dim, 2*latent_dim)
        self.encoder_fc_3 = nn.Linear(2*latent_dim, latent_dim)
        self.relu = nn.ReLU()

        ## DECODER
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),
            nn.ReLU(),
            nn.Linear(2*latent_dim, 4*latent_dim),
            nn.ReLU(),
            nn.Linear(4*latent_dim, n_features),
            nn.ReLU(),
        )

    def forward(self, x):
        # Aplicar el positional encoding a las entradas
        x = self.positional_encoding(x)

        # Aplicar atención multi-cabezal
        attn_output, _ = self.attention(x, x, x)

        # Normalización
        attn_output = self.layer_norm(attn_output)

        # Capas densas y activación
        encoded_1 = self.relu(self.encoder_fc_1(attn_output))
        encoded_2 = self.relu(self.encoder_fc_2(encoded_1))
        encoded_3 = self.relu(self.encoder_fc_3(encoded_2))
        
        # Decodificación
        decoded = self.decoder(encoded_3)
        return decoded
    
    def predict(self, threshold_list, df, window_size, batch_size, device):
        generator = sequence_generator(df.values, window_size, batch_size)
        steps = (len(df) - window_size) // batch_size

        # Generar las secuencias
        anomalies = []
        for _ in tqdm(range(steps), desc="Detectando anomalías"):
            batch, _ = next(generator)  # Ignorar las etiquetas
            reconstructed_batch = self.forward(batch.to(device)).detach().cpu().numpy()
            batch = batch.cpu().numpy()  # Convert to numpy array once

            # Calculate reconstruction error
            reconstruction_error = np.square(batch - reconstructed_batch)
            last_elements = reconstruction_error[:, -1, :]
            
            # Use numpy array to avoid slow tensor conversion
            anomalies.extend((last_elements > threshold_list).astype(int).tolist())

        # Convert to DataFrame
        first_el = window_size - 1
        last_el = first_el + len(anomalies)
        return pd.DataFrame(anomalies, columns=df.columns, index=df.index[first_el:last_el])