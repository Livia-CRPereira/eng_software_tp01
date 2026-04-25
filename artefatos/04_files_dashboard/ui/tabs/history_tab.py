"""
Aba de histórico de exames salvos.

Princípios aplicados:
- Responsabilidade Única: renderiza apenas o histórico; sub-responsabilidades delegadas a métodos privados.
- Prefira Composição a Herança: recebe colaboradores via injeção de dependência.
- Lei de Demeter: acessa apenas métodos públicos de repository e visualizer;
  não desce em sub-objetos internos.
- Inversão de Dependências: depende de IExamRepository, não da implementação JSON.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import ANOMALY_CLASSES
from domain.interfaces import IExamRepository
from ui.visualizer import SpectrogramVisualizer


class HistoryTab:
    """Responsável pela renderização da aba de histórico de exames."""

    def __init__(self, repository: IExamRepository, visualizer: SpectrogramVisualizer) -> None:
        self._repository = repository
        self._visualizer = visualizer

    def render(self) -> None:
        st.subheader("Histórico de Exames salvos")
        history = self._repository.load_all()

        if not history:
            st.info("Nenhum paciente salvo ainda.")
            return

        self._render_overview(history)
        st.divider()
        self._render_details(history)

    # -- Métodos privados (Ocultamento de Informação) --

    def _render_overview(self, history: dict) -> None:
        st.subheader("Visão Geral dos Exames")

        ids_exams    = list(history.keys())
        probs_matrix = [data['Probabilidades'] for data in history.values()]
        main_results = [ANOMALY_CLASSES[np.argmax(p)] for p in probs_matrix]

        df_count      = pd.Series(main_results).value_counts()
        common_anomaly = df_count.index[0]

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total de Exames Salvos", len(ids_exams))
        col_m2.metric("Anomalia mais frequente", common_anomaly)

        col_graf1, col_graf2 = st.columns(2)

        with col_graf1:
            st.write("**Distribuição de Diagnósticos Principais**")
            st.bar_chart(df_count)

        with col_graf2:
            st.write("**Mapa de Calor de Risco por Paciente**")
            self._render_heatmap(probs_matrix, ids_exams)

    def _render_heatmap(self, probs_matrix: list, ids_exams: list) -> None:
        fig_hm, ax_hm = plt.subplots(figsize=(6, 4))
        cax_hm = ax_hm.imshow(probs_matrix, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        fig_hm.colorbar(cax_hm, ax=ax_hm, label='Probabilidade')

        ax_hm.set_xticks(range(6))
        ax_hm.set_xticklabels(ANOMALY_CLASSES, rotation=45, ha="right")
        ax_hm.set_yticks(range(len(ids_exams)))
        ax_hm.set_yticklabels(ids_exams)

        st.pyplot(fig_hm)

    def _render_details(self, history: dict) -> None:
        st.subheader("Detalhamento de Exames")

        for id_exam, data in history.items():
            with st.expander(f"Exame: {id_exam}"):
                col_data, col_chart = st.columns([2, 1])

                with col_data:
                    st.write("**Probabilidades de Anomalia:**")
                    df_resume = pd.DataFrame({
                        "Classe": ANOMALY_CLASSES,
                        "Probabilidade": [f"{p * 100:.2f}%" for p in data['Probabilidades']],
                    })
                    st.table(df_resume)

                with col_chart:
                    matrix = np.array([data['Espectrograma']])
                    st.pyplot(self._visualizer.plot(matrix, title=f'Espectrograma: {id_exam}'))

                if st.button(f"Remover {id_exam}", key=f'del_{id_exam}'):
                    if self._repository.delete(id_exam):
                        st.toast(f'Exame {id_exam} removido')
                        time.sleep(1)
                        st.rerun()
