"""
Implementação concreta do repositório de exames em JSON.

Princípios aplicados:
- Responsabilidade Única: responsável exclusivamente pela leitura/escrita do arquivo JSON.
- Ocultamento de Informação: detalhes do formato de persistência (JSON, caminhos) ficam encapsulados aqui.
- Inversão de Dependências: implementa IExamRepository; o restante do sistema depende da interface,
  não desta classe diretamente.
- Substituição de Liskov: pode ser trocada por outra implementação (SQLite, Redis, etc.)
  sem alterar o código consumidor.
"""

import json
import os
from typing import Dict

from config import DB_FILE
from domain.entities import ExamResult
from domain.interfaces import IExamRepository


class JsonExamRepository(IExamRepository):
    """Persiste exames em um arquivo JSON local."""

    def __init__(self, filepath: str = DB_FILE) -> None:
        self._filepath = filepath

    # -- Métodos privados de I/O (Ocultamento de Informação) --

    def _load_raw(self) -> dict:
        if os.path.exists(self._filepath):
            with open(self._filepath, "r") as f:
                return json.load(f)
        return {}

    def _save_raw(self, data: dict) -> None:
        with open(self._filepath, "w") as f:
            json.dump(data, f)

    # -- Contrato público (IExamRepository) --

    def save(self, result: ExamResult) -> None:
        data = self._load_raw()
        data[result.exam_id] = {
            "Probabilidades": result.probabilities,
            "Espectrograma": result.spectrogram,
        }
        self._save_raw(data)

    def load_all(self) -> Dict[str, dict]:
        return self._load_raw()

    def delete(self, exam_id: str) -> bool:
        data = self._load_raw()
        if exam_id in data:
            del data[exam_id]
            self._save_raw(data)
            return True
        return False

    def exists(self, exam_id: str) -> bool:
        return exam_id in self._load_raw()
