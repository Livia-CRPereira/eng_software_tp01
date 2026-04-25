"""
Aba de conversão de arquivos.

Princípios aplicados:
- Responsabilidade Única: renderiza apenas a aba de conversão de formato.
- Prefira Composição a Herança: recebe o conversor via injeção de dependência.
"""

import streamlit as st

from services.spectrogram_converter import SpectrogramConverter


class ConverterTab:
    """Responsável pela renderização da aba de conversão Parquet → NumPy."""

    def __init__(self, converter: SpectrogramConverter) -> None:
        self._converter = converter

    def render(self) -> None:
        st.subheader("Conversor de Espectrograma (.parquet para .npy)")

        arquivo_parquet = st.file_uploader(
            "Faça o upload do espectrograma bruto (.parquet)", type=["parquet"]
        )

        if arquivo_parquet is not None:
            st.info("Lendo arquivo bruto...")
            buffer = self._converter.parquet_to_npy_buffer(arquivo_parquet)
            st.success("Conversão finalizada com sucesso!")

            st.download_button(
                label="📥 Baixar arquivo .npy",
                data=buffer,
                file_name="espectrograma_pronto.npy",
                mime="application/octet-stream",
            )
