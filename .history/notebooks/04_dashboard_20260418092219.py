import streamlit as st
import numpy as np
import xgboost as xgb
import os

# ---------------------------------------------------------
# 1. CARREGAMENTO DOS MODELOS (CACHE)
# ---------------------------------------------------------
# st.cache_resource garante que os modelos só sejam lidos do disco uma vez, poupando tempo.
@st.cache_resource
def carregar_modelos():
    pasta_modelos = 'modelos_treinados' # Ajuste se o caminho da sua pasta for outro
    modelos = []
    
    for i in range(5):
        model = xgb.XGBClassifier()
        caminho_arquivo = os.path.join(pasta_modelos, f'xgboost_fold_{i}.json')
        model.load_model(caminho_arquivo)
        modelos.append(model)
        
    return modelos

# ---------------------------------------------------------
# 2. EXTRAÇÃO DE FEATURES (Versão 2 - Dinâmica Temporal)
# ---------------------------------------------------------
def extrair_features_paciente(arquivo_upload):
    # Carrega o arquivo .npy enviado pelo Streamlit
    img = np.load(arquivo_upload)

    # Estatísticas Globais
    mean_global = np.nanmean(img, axis=0)
    std_global = np.nanstd(img, axis=0)
    max_global = np.nanmax(img, axis=0)
    min_global = np.nanmin(img, axis=0)

    # Dinâmica Temporal (Fatiamento em 3 janelas)
    mean_t1 = np.nanmean(img[:100, :], axis=0)
    mean_t2 = np.nanmean(img[100:200, :], axis=0)
    mean_t3 = np.nanmean(img[200:, :], axis=0)

    # Variância Local
    std_t1 = np.nanstd(img[:100, :], axis=0)
    std_t3 = np.nanstd(img[200:, :], axis=0)

    # Vetor final de 3.600 características
    feature_vector = np.concatenate([
        mean_global, std_global, max_global, min_global,
        mean_t1, mean_t2, mean_t3, std_t1, std_t3
    ])

    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    
    # Transforma o vetor (3600,) em uma matriz de uma linha (1, 3600) exigida pelo XGBoost
    return feature_vector.reshape(1, -1)

# ---------------------------------------------------------
# 3. INTERFACE BÁSICA (A ser melhorada pela equipe)
# ---------------------------------------------------------
st.title("🧠 MVP: Triagem de EEG (Back-end)")
st.write("Insira o espectrograma do paciente para obter o cálculo do Ensemble.")

# Inicia os modelos
modelos_ensemble = carregar_modelos()

# Área de Upload
arquivo = st.file_uploader("Upload do Espectrograma (.npy)", type=["npy"])

if arquivo is not None:
    st.info("Arquivo recebido! Iniciando extração e inferência...")
    
    # Processa o arquivo gerando as 3600 características
    X_paciente = extrair_features_paciente(arquivo)
    
    # Predição com o Ensemble (Junta Médica)
    probabilidades_acumuladas = np.zeros(6)
    
    for modelo in modelos_ensemble:
        # Pega a primeira linha de predições já que temos só 1 paciente
        probabilidades_acumuladas += modelo.predict_proba(X_paciente)[0]
        
    # Tira a média dos 5 modelos
    probabilidade_final = probabilidades_acumuladas / 5.0
    
    # ---------------------------------------------------------
    # 4. SAÍDA DE DADOS (Output)
    # ---------------------------------------------------------
    st.success("Inferência Concluída!")
    
    # Dicionário mapeando os resultados para as 6 classes
    classes = ['Seizure (Convulsão)', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other (Outros)']
    
    # Mostra de forma crua, só para garantir a matemática. O Colega 2 vai deixar isso bonito!
    st.subheader("Probabilidades Calculadas:")
    for nome_classe, prob in zip(classes, probabilidade_final):
        # Multiplica por 100 para mostrar em formato de porcentagem
        st.write(f"**{nome_classe}:** {prob * 100:.2f}%")