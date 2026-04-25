"""
Carregamento dos modelos XGBoost treinados.

Princípios aplicados:
- Responsabilidade Única: responsável exclusivamente por carregar e cachear modelos do disco.
- Ocultamento de Informação: o caminho e a lógica de carregamento são isolados aqui.
"""

import os
from typing import List

import streamlit as st
import xgboost as xgb

from config import MODEL_DIR, NUM_FOLDS


@st.cache_resource
def load_xgboost_models() -> List[xgb.XGBClassifier]:
    """Carrega os modelos XGBoost de todos os folds. Resultado é cacheado pelo Streamlit."""
    models = []
    for i in range(NUM_FOLDS):
        model = xgb.XGBClassifier()
        file_path = os.path.join(MODEL_DIR, f'xgboost_fold_{i}.json')
        model.load_model(file_path)
        models.append(model)
    return models
