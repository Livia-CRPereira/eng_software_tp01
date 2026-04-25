"""
Ponto de entrada da aplicação Streamlit.

Princípios aplicados:
- Responsabilidade Única: apenas monta a UI e compõe as dependências.
- Prefira Composição a Herança: toda a aplicação é construída pela composição de objetos colaboradores.
- Inversão de Dependências: este módulo é a 'raiz de composição' (Composition Root);
  instancia as implementações concretas e as injeta nas abstrações que as consomem.
  Nenhum outro módulo instancia dependências diretamente.
- Acoplamento Baixo: alterações em implementações concretas (ex: trocar JSON por banco de dados)
  impactam apenas este arquivo.
"""

import streamlit as st

from config import ANOMALY_GLOSSARY, DB_FILE
from infrastructure.exam_repository import JsonExamRepository
from infrastructure.model_loader import load_xgboost_models
from services.ensemble_predictor import EnsemblePredictor
from services.feature_extractor import SpectrogramFeatureExtractor
from services.spectrogram_converter import SpectrogramConverter
from ui.tabs.converter_tab import ConverterTab
from ui.tabs.diagnosis_tab import DiagnosisTab
from ui.tabs.history_tab import HistoryTab
from ui.visualizer import SpectrogramVisualizer

# ---------------------------------------------------------------------------
# Composição de dependências (Composition Root)
# Implementações concretas são instanciadas aqui e injetadas via interface.
# ---------------------------------------------------------------------------

repository  = JsonExamRepository(filepath=DB_FILE)
models      = load_xgboost_models()
extractor   = SpectrogramFeatureExtractor()
predictor   = EnsemblePredictor(models=models)
converter   = SpectrogramConverter()
visualizer  = SpectrogramVisualizer()

diagnosis_tab_ui = DiagnosisTab(repository, extractor, predictor, visualizer)
converter_tab_ui = ConverterTab(converter)
history_tab_ui   = HistoryTab(repository, visualizer)

# ---------------------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------------------

st.title("🧠 MVP: Sistema de Apoio de Diagnóstico")

diagnosis_tab, conversor_tab, history_tab = st.tabs([
    "🩺 Triagem (Diagnóstico)",
    "⚙️ Conversor de Arquivos",
    "📜 Pacientes Salvos",
])

with st.sidebar:
    st.header("Centro de Ajuda")
    with st.popover("Glossário de Anomalias"):
        st.markdown("### Definições das Anomalias")
        for term, definition in ANOMALY_GLOSSARY.items():
            st.markdown(f"* **{term}:** {definition}")
    st.divider()

with diagnosis_tab:
    diagnosis_tab_ui.render()

with conversor_tab:
    converter_tab_ui.render()

with history_tab:
    history_tab_ui.render()
