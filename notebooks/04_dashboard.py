import streamlit as st
import numpy as np
import xgboost as xgb
import os
import pandas as pd
import io
import json
import matplotlib.pyplot as plt
import time

DB_FILE = 'exams_saved.json'

def load_history():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_history(id_exam, probs, spectro_matrix):
    history = load_history()

    history[id_exam] = {
        "Probabilidades": probs.tolist(),
        "Espectrograma": spectro_matrix.tolist()
    }

    with open(DB_FILE, "w") as f:
        json.dump(history, f)

def delete_history(id_exam):
    history = load_history()
    if id_exam in history:
        del history[id_exam]
        with open(DB_FILE, 'w') as f:
            json.dump(history, f)
        return True
    return False

def exam_exists(id_exam):
    history = load_history()
    return id_exam in history

# ---------------------------
# 1. CARREGAMENTO DOS MODELOS
# ---------------------------
@st.cache_resource
def carregar_modelos():
    pasta_modelos = 'modelos_treinados'
    modelos = []
    
    for i in range(5):
        model = xgb.XGBClassifier()
        caminho_arquivo = os.path.join(pasta_modelos, f'xgboost_fold_{i}.json')
        model.load_model(caminho_arquivo)
        modelos.append(model)
        
    return modelos

# -----------------------
# 2. EXTRAÇÃO DE FEATURES
# -----------------------
def extrair_features_paciente(img):

    mean_global = np.nanmean(img, axis=0)
    std_global = np.nanstd(img, axis=0)
    max_global = np.nanmax(img, axis=0)
    min_global = np.nanmin(img, axis=0)

    mean_t1 = np.nanmean(img[:100, :], axis=0)
    mean_t2 = np.nanmean(img[100:200, :], axis=0)
    mean_t3 = np.nanmean(img[200:, :], axis=0)

    std_t1 = np.nanstd(img[:100, :], axis=0)
    std_t3 = np.nanstd(img[200:, :], axis=0)

    feature_vector = np.concatenate([
        mean_global, std_global, max_global, min_global,
        mean_t1, mean_t2, mean_t3, std_t1, std_t3
    ])

    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    
    return feature_vector.reshape(1, -1)

def plot_spectro(matrix, title='Espectrograma do Paciente'):
    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.imshow(matrix.T, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Magnitude')

    ax.set_title(title)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Frequências")

    return fig

# -------------------
# 3. INTERFACE BÁSICA 
# -------------------
st.title("🧠 MVP: Sistema de Apoio de Diagnóstico")
aba_triagem, aba_conversor, aba_historico = st.tabs(["🩺 Triagem (Diagnóstico)", "⚙️ Conversor de Arquivos", "📜 Pacientes Salvos"])

with aba_triagem:
    st.subheader("Análise de Paciente")
    st.write("Insira o espectrograma do paciente para obter o cálculo do Ensemble.")

    modelos_ensemble = carregar_modelos()
    arquivo = st.file_uploader("Upload do Espectrograma (.npy)", type=["npy"])

    if arquivo is not None:
        st.info("Arquivo recebido! Iniciando extração e inferência...")
        
        img_paciente = np.load(arquivo)

        X_paciente = extrair_features_paciente(img_paciente)
        
        probabilidades_acumuladas = np.zeros(6)
        
        for modelo in modelos_ensemble:
            probabilidades_acumuladas += modelo.predict_proba(X_paciente)[0]
            
        probabilidade_final = probabilidades_acumuladas / len(modelos_ensemble)
        
        # ---------------------
        # 4. SAÍDA DO RESULTADO
        # ---------------------
        st.success("Inferência Concluída!")
        
        classes = ['Seizure (Convulsão)', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other (Outros)']
        
        st.subheader("Probabilidades Calculadas:")
        for nome_classe, prob in zip(classes, probabilidade_final):
            st.write(f"**{nome_classe}:** {prob * 100:.2f}%")

        # -------------------
        # 5. SAÍDA DO GRÁFICO
        # -------------------

        st.pyplot(plot_spectro(img_paciente))

        st.divider()
        st.subheader("💾 Salvar Resultado")
        id_input = st.text_input("Identificador do Paciente/Exame (ex: PAC-123)")

        if st.button("Confirmar e Salvar"):
            if not id_input:
                st.error("Insira um ID para salvar")
            elif exam_exists(id_input):
                st.warning(f"Erro: O ID '{id_input}' já está cadastrado, use outro ou exclua o antigo")
            else:
                save_history(id_input, probabilidade_final, img_paciente)
                st.success(f"Paciente {id_input} salvo com sucesso!")

with aba_conversor:
    st.subheader("Conversor de Espectrograma (.parquet para .npy)")

    arquivo_parquet = st.file_uploader("Faça o upload do espectrograma bruto (.parquet)", type=["parquet"])

    if arquivo_parquet is not None:
        st.info("Lendo arquivo bruto...")
        
        df_espectro = pd.read_parquet(arquivo_parquet)
        
        if 'time' in df_espectro.columns:
            df_espectro = df_espectro.drop(columns=['time'])
            
        matriz_npy = df_espectro.fillna(0).values
        
        buffer_memoria = io.BytesIO()
        np.save(buffer_memoria, matriz_npy)
        buffer_memoria.seek(0)
        
        st.success("Conversão finalizada com sucesso!")
        
        st.download_button(
            label="📥 Baixar arquivo .npy",
            data=buffer_memoria,
            file_name="espectrograma_pronto.npy",
            mime="application/octet-stream"
        )

with aba_historico:
    st.subheader("Histórico de Exames salvos")
    historico = load_history()

    if not historico:
        st.info("Nenhum paciente salvo ainda.")
    else:
        classes = ['Seizure (Convulsão)', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other (Outros)']

        for id_exam, data in historico.items():
            with st.expander(f"Exame: {id_exam}"):
                col_data, col_chart = st.columns([2, 1])
                
                with col_data:
                    st.write("**Probabilidades de Anomalia:**")
                    df_resume = pd.DataFrame({
                        "Classe": classes, 
                        "Probabilidade": [f"{p*100:.2f}%" for p in data['Probabilidades']]
                    })

                    st.table(df_resume)

                with col_chart:
                    matrix = np.array([data['Espectrograma']])
                    st.pyplot(plot_spectro(matrix, title=f'Espectrograma: {id_exam}'))


                if st.button(f"Remover {id_exam}", key=f'del_{id_exam}'):
                    if delete_history(id_exam):
                        st.toast(f'Exame {id_exam} removido')

                        time.sleep(1)

                        st.rerun()