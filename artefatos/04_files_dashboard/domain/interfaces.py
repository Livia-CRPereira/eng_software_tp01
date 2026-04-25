"""
Contratos (interfaces) do domínio da aplicação.

Princípios aplicados:
- Segregação de Interfaces: cada interface tem responsabilidade mínima e focada.
- Inversão de Dependências: módulos de alto nível dependem dessas abstrações, não de implementações concretas.
- Substituição de Liskov: qualquer implementação concreta pode substituir a interface sem quebrar o sistema.
- Aberto/Fechado: novas implementações (ex: outro banco de dados, outro modelo) podem ser adicionadas
  sem alterar o código que as consome.
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from domain.entities import ExamResult


class IExamRepository(ABC):
    """Contrato para persistência de exames."""

    @abstractmethod
    def save(self, result: ExamResult) -> None:
        """Persiste um resultado de exame."""
        ...

    @abstractmethod
    def load_all(self) -> Dict[str, dict]:
        """Retorna todos os exames salvos."""
        ...

    @abstractmethod
    def delete(self, exam_id: str) -> bool:
        """Remove um exame pelo ID. Retorna True se removido com sucesso."""
        ...

    @abstractmethod
    def exists(self, exam_id: str) -> bool:
        """Verifica se um exame com o ID fornecido já existe."""
        ...


class IFeatureExtractor(ABC):
    """Contrato para extração de features de um espectrograma."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extrai um vetor de features a partir de uma imagem/matriz."""
        ...


class IPredictor(ABC):
    """Contrato para predição de probabilidades de anomalia."""

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Retorna um vetor de probabilidades por classe."""
        ...
