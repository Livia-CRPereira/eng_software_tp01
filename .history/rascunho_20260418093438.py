import numpy as np
import pandas as pd

# 1. Carrega o arquivo .npy do paciente de teste (ajuste o caminho se necessário)
matriz = np.load('data/spectrogram_processado_mvp/1544207007.npy')

# 2. Transforma a matriz (que tem 400 colunas) em um DataFrame do Pandas
df = pd.DataFrame(matriz)

# 3. Adiciona uma coluna 'time' falsa só para o seu conversor do Streamlit poder dropar ela depois
# (Vamos colocar uma sequência de números simples fingindo ser os segundos)
df['time'] = np.arange(len(df))

# 4. Salva o resultado no formato .parquet
df.to_parquet('exame_bruto_teste.parquet', index=False)

print("✅ Arquivo 'exame_bruto_teste.parquet' gerado com sucesso!")