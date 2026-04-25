"""
Módulo de configuração central da aplicação.
Centraliza todas as constantes e evita magic strings/numbers espalhados pelo código.
"""

DB_FILE: str = 'exams_saved.json'
MODEL_DIR: str = 'modelos_treinados'
NUM_FOLDS: int = 5

ANOMALY_CLASSES: list[str] = [
    'Seizure (Convulsão)',
    'LPD',
    'GPD',
    'LRDA',
    'GRDA',
    'Other (Outros)',
]

ANOMALY_GLOSSARY: dict[str, str] = {
    "Seizure": "Convulsão",
    "LPD": "Descargas Periódicas Lateralizadas",
    "GPD": "Descargas Periódicas Generalizadas",
    "LRDA": "Atividade Delta Rítmica Lateralizada",
    "GRDA": "Atividade Delta Rítmica Generalizada",
    "Other": "Outros (Padrões que não se encaixam nas anomalias especificadas)",
}
