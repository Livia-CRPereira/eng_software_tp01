"""
Entidades de domínio da aplicação.

Princípios aplicados:
- Coesão: dados relacionados agrupados em uma única estrutura coesa.
- Ocultamento de Informação: a estrutura interna é definida aqui e não
  replicada como dicionários soltos ao longo do código.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ExamResult:
    """Representa o resultado completo de um exame processado."""

    exam_id: str
    probabilities: List[float]
    spectrogram: List[List[float]]
