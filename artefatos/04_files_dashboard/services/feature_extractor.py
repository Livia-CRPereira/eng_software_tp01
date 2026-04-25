"""
Extração de features do espectrograma.

Princípios aplicados:
- Responsabilidade Única: responsável exclusivamente por transformar uma matriz numpy em vetor de features.
- Inversão de Dependências: implementa IFeatureExtractor, permitindo substituição por outro extrator.
- Substituição de Liskov / Aberto-Fechado: novas estratégias de extração podem ser criadas
  implementando IFeatureExtractor sem alterar o código consumidor.
"""

import numpy as np

from domain.interfaces import IFeatureExtractor


class SpectrogramFeatureExtractor(IFeatureExtractor):
    """Extrai estatísticas temporais e globais de um espectrograma."""

    def extract(self, image: np.ndarray) -> np.ndarray:
        mean_global = np.nanmean(image, axis=0)
        std_global  = np.nanstd(image,  axis=0)
        max_global  = np.nanmax(image,  axis=0)
        min_global  = np.nanmin(image,  axis=0)

        mean_t1 = np.nanmean(image[:100,  :], axis=0)
        mean_t2 = np.nanmean(image[100:200, :], axis=0)
        mean_t3 = np.nanmean(image[200:,  :], axis=0)

        std_t1 = np.nanstd(image[:100, :], axis=0)
        std_t3 = np.nanstd(image[200:, :], axis=0)

        feature_vector = np.concatenate([
            mean_global, std_global, max_global, min_global,
            mean_t1, mean_t2, mean_t3, std_t1, std_t3,
        ])

        return np.nan_to_num(feature_vector, nan=0.0).reshape(1, -1)
