"""
Predição por ensemble de modelos.

Princípios aplicados:
- Responsabilidade Única: responsável exclusivamente pela lógica de agregação do ensemble.
- Prefira Composição a Herança: recebe a lista de modelos via injeção (composição),
  em vez de herdar de uma classe base de modelo.
- Inversão de Dependências: implementa IPredictor; consome modelos via duck-typing,
  não acoplado à classe XGBClassifier diretamente.
- Lei de Demeter: EnsemblePredictor só chama predict_proba nos modelos que recebe;
  não acessa estruturas internas deles.
"""

from typing import List

import numpy as np

from domain.interfaces import IPredictor


class EnsemblePredictor(IPredictor):
    """Agrega predições de múltiplos modelos pela média das probabilidades."""

    def __init__(self, models: List) -> None:
        self._models = models

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        cumulative = np.zeros(6)
        for model in self._models:
            cumulative += model.predict_proba(features)[0]
        return cumulative / len(self._models)
