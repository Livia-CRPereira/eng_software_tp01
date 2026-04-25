"""
Conversão de espectrogramas de .parquet para .npy.

Princípios aplicados:
- Responsabilidade Única: responsável exclusivamente pela conversão de formato de arquivo.
- Ocultamento de Informação: a lógica de limpeza de colunas e serialização fica encapsulada aqui.
"""

import io

import numpy as np
import pandas as pd


class SpectrogramConverter:
    """Converte espectrogramas do formato Parquet para o formato NumPy (.npy)."""

    def parquet_to_npy_buffer(self, file) -> io.BytesIO:
        """Lê um arquivo Parquet e retorna um buffer .npy pronto para download."""
        df = pd.read_parquet(file)

        if 'time' in df.columns:
            df = df.drop(columns=['time'])

        matrix = df.fillna(0).values

        buffer = io.BytesIO()
        np.save(buffer, matrix)
        buffer.seek(0)

        return buffer
