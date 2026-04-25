"""
Visualização de espectrogramas.

Princípios aplicados:
- Responsabilidade Única: responsável exclusivamente pela geração de figuras matplotlib.
- Ocultamento de Informação: configurações visuais (cmap, labels, tamanho) ficam encapsuladas aqui.
"""

import matplotlib.pyplot as plt
import numpy as np


class SpectrogramVisualizer:
    """Gera figuras matplotlib de espectrogramas."""

    def plot(self, matrix: np.ndarray, title: str = 'Espectrograma do Paciente') -> plt.Figure:
        """Retorna uma Figure matplotlib com o espectrograma plotado."""
        fig, ax = plt.subplots(figsize=(10, 4))

        cax = ax.imshow(matrix.T, aspect='auto', origin='lower', cmap='viridis')
        fig.colorbar(cax, ax=ax, label='Magnitude')

        ax.set_title(title)
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Frequências")

        return fig
