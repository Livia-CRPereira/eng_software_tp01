# Detecção de Atividade Cerebral Prejudicial

## Objetivos e Principais Features

Atualmente, profissionais da saúde detectam atividades cerebrais prejudiciais com análise manual de eletroencefalografia (EEG), exame realizado em pacientes em estado crítico. Esse processo, porém, está propenso a erros relacionados à fadiga e problemas de confiabilidade entre revisores, além de ser um processo caro. Dessa forma, o objetivo desse projeto é desenvolver um modelo que melhore a precisão de classificação e detecção de padrões da atividade cerebral a partir da EEG, contribuindo para um diagnóstico mais rápido de possíveis problemas e, consequentemente, tratamentos mais eficazes. O modelo será treinado com dados de EEG, utilizando anotações de especialistas como "ground truth". 

A hipótese principal se molda sob a perspectiva de que a utilização dos arquivos de EEG, que capturam as variações elétricas de curtíssimo prazo, juntamente às frequências do cérebro registradas pelos espectogramas, que registram o comportamento das frequências do cérebro em janelas maiores de tempo, consigam ser uma boa base para treinar o modelo, permitindo uma construção de modelo que identifique corretamente padrões de atividades prejudiciais no cérebro, identificando problemas como convulsões, derrames e paradas cardíacas de forma precoce. 

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
