"""
Aba de diagnóstico do sistema.

Princípios aplicados:
- Responsabilidade Única: renderiza apenas a aba de triagem/diagnóstico.
- Prefira Composição a Herança: recebe colaboradores via construtor (injeção de dependências).
- Lei de Demeter: só interage com suas dependências diretas; não acessa sub-objetos internos delas.
- Inversão de Dependências: depende das interfaces IExamRepository, IFeatureExtractor e IPredictor,
  não de implementações concretas.
"""

import numpy as np
import streamlit as st

from config import ANOMALY_CLASSES
from domain.entities import ExamResult
from domain.interfaces import IExamRepository, IFeatureExtractor, IPredictor
from ui.visualizer import SpectrogramVisualizer


class DiagnosisTab:
    """Responsável pela renderização da aba de análise e diagnóstico de pacientes."""

    def __init__(
        self,
        repository: IExamRepository,
        extractor: IFeatureExtractor,
        predictor: IPredictor,
        visualizer: SpectrogramVisualizer,
    ) -> None:
        self._repository = repository
        self._extractor = extractor
        self._predictor = predictor
        self._visualizer = visualizer

    def render(self) -> None:
        st.subheader("Análise de Paciente")
        st.write("Insira o espectrograma do paciente para obter o cálculo do Ensemble.")

        file = st.file_uploader("Upload do Espectrograma (.npy)", type=["npy"])

        if file is not None:
            self._process_file(file)

    # -- Métodos privados (Ocultamento de Informação) --

    def _process_file(self, file) -> None:
        st.info("Arquivo recebido! Iniciando extração e inferência...")

        exam_img = np.load(file)
        features = self._extractor.extract(exam_img)
        final_prob = self._predictor.predict_proba(features)

        st.success("Inferência Concluída!")
        self._render_probabilities(final_prob)
        st.pyplot(self._visualizer.plot(exam_img))
        st.divider()
        self._render_save_section(final_prob, exam_img)

    def _render_probabilities(self, final_prob: np.ndarray) -> None:
        st.subheader("Probabilidades Calculadas:")
        for class_name, prob in zip(ANOMALY_CLASSES, final_prob):
            st.write(f"**{class_name}:** {prob * 100:.2f}%")

    def _render_save_section(self, final_prob: np.ndarray, exam_img: np.ndarray) -> None:
        st.subheader("💾 Salvar Resultado")
        id_input = st.text_input("Identificador do Paciente/Exame (ex: PAC-123)")

        if st.button("Confirmar e Salvar"):
            self._handle_save(id_input, final_prob, exam_img)

    def _handle_save(self, exam_id: str, final_prob: np.ndarray, exam_img: np.ndarray) -> None:
        if not exam_id:
            st.error("Insira um ID para salvar")
        elif self._repository.exists(exam_id):
            st.warning(f"Erro: O ID '{exam_id}' já está cadastrado, use outro ou exclua o antigo")
        else:
            result = ExamResult(
                exam_id=exam_id,
                probabilities=final_prob.tolist(),
                spectrogram=exam_img.tolist(),
            )
            self._repository.save(result)
            st.success(f"Paciente {exam_id} salvo com sucesso!")
