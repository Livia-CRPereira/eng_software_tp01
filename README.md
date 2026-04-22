# Detecção de Atividade Cerebral Prejudicial

## Objetivos e Principais Features

Atualmente, a detecção de atividades cerebrais prejudiciais em pacientes críticos é feita, em grande parte, por análise manual de exames de eletroencefalografia (EEG), um processo sujeito a erros causados por fadiga, divergência entre especialistas e alto custo. Este projeto tem como objetivo desenvolver um modelo que melhore a precisão na classificação e detecção de padrões cerebrais a partir de dados de EEG, contribuindo para diagnósticos mais rápidos e tratamentos mais eficazes.

 A hipótese principal é que a combinação dos sinais brutos de EEG, que registram variações elétricas de curto prazo, com espectrogramas, que representam o comportamento das frequências cerebrais em janelas maiores de tempo, fornece uma base eficiente para treinar um modelo capaz de identificar precocemente problemas como convulsões, derrames e paradas cardíacas.


### Dataset

**Origem:** [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview)

A base é composta por metadados (train.csv e test.csv) que guiam para os dados brutos separados em diretórios. Os dados brutos contém trechos de 50 segundos de EEG (amostrados a 200 Hz) e espectrogramas cobrindo janelas de 10 minutos de 1.950 pacientes únicos.

* Metadados (train.csv): Possui 15 colunas, sendo 8 colunas de identificação e mapeamento, 1 coluna de texto com o consenso dos especialistas e 6 colunas de Target (votos numéricos para cada classe de anomalia).
    
* Features de Entrada (EEG Bruto - .parquet): Matrizes de 10.000 linhas × 20 colunas, representando o avanço do tempo *versus* a voltagem captada pelos canais de eletrodos cerebrais e um eletrocardiograma (EKG).
    
* Features de Entrada (Espectrogramas - .parquet): Matrizes de 300 linhas × 401 colunas, representando recortes de tempo *versus* frequências em Hertz divididas pelas regiões do cérebro.
      
## Membros da Equipe e Papéis

*   João Pedro Moreira Smolinski: Cientista de Dados + Analista de Dados (EDA, Visualização dos dados e apoio à modelagem)
*   Letícia Ruas de Lucca Rodrigues: Cientista de Dados (Validação) (Treinamento, validação e otimização de modelos)
*   Lívia Caroline Rodrigues Pereira: Cientista de Dados + Engenheira de Dados (Coleta, limpeza, tratamento de dados e preparação de pipelines)

## Pilha de Tecnologias

* Linguagem de Programação Python
* Bibliotecas de visualização, manipulação e análise de dados: Pandas, NumPy, SciPy, Matplotlib, Seaborn
* Bibliotecas de modelagem: PyTorch
* Modelos de Aprendizado de Máquina Transformers
* Ambiente de desenvolvimento: GitHub, Jupyter Notebook, VS Code

## Especificações do Problema

Como o problema aqui proposto envolve medicina, uma área específica e de conhecimento técnico e nichado, essa seção apresenta alguns conceitos importantes para os desenvolvedores e para os interessados na análise. 

1. A Captação (Eletrodos): Quando um paciente crítico necessita de monitoramento, sensores de metal (eletrodos) são fixados em regiões específicas do seu couro cabeludo (como os lobos frontal, temporal, parietal e occipital).

2. O Sinal Bruto (EEG): Esses eletrodos medem a diferença de voltagem elétrica gerada pela comunicação dos neurônios ao longo do tempo. Esse registro contínuo e em alta frequência (200 medições por segundo) gera as matrizes de Séries Temporais (EEG Bruto), onde picos agudos podem indicar convulsões ou lesões localizadas.

3. A Visão de Frequência (Espectrograma): Como ler horas de ondas elétricas brutas é inviável, o sinal de EEG é matematicamente transformado em Espectrogramas. Eles funcionam como "mapas de calor", mostrando quais frequências (ondas lentas ou rápidas) estão mais fortes em um período de 10 minutos, ajudando a identificar lentidão cerebral que indica intoxicações ou danos graves.

4. A Avaliação Humana (Target): Neurologistas analisam essas duas representações visuais em busca de padrões anômalos (como LPD, GPD, LRDA). Como a interpretação dessas ondas é altamente complexa e subjetiva, especialistas frequentemente discordam do diagnóstico. Por isso, a nossa variável alvo não é uma classe única, mas sim uma distribuição de probabilidade baseada na votação de múltiplos médicos, refletindo o consenso (ou a incerteza) real da medicina.
