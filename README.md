# Rede Neural com predições de volume útil do reservatório de Funil (RJ) baseado em variáveis climáticas

_Trabalho de Conclusão de Curso apresentado ao Curso de Especialização em Inteligência Artificial e Aprendizado de Máquina da Pontifícia Universidade Católica de Minas Gerais._  

**Autor:** Marina Micas Jardim  
**Data:** Abril de 2022

Este projeto tem como objetivo prever, utilizando uma Rede Neural Feed-Forward (FFNN), o volume útil do Reservatório de Funil (RJ) com base em variáveis climáticas. A iniciativa integra dados meteorológicos e hidrológicos, permitindo análises que auxiliem no planejamento da utilização do recurso hídrico, evitando, por exemplo, a exploração inadequada do volume morto. O trabalho se apoia em técnicas de pré-processamento, análise exploratória, modelagem com deep learning e avaliação via métricas estatísticas.

## Sumário

1. 😁​ [Introdução](#introdução)
2. 😀​ [Contextualização](#contextualização)
3. 😏​​ [Descrição do Problema e Solução Proposta](#descrição-do-problema-e-solução-proposta)
4. 😅​​ [Coleta de Dados](#coleta-de-dados)
   - [Dados da Estação Meteorológica de Resende (RJ)](#dados-da-estação-meteorológica-de-resende-rj)
   - [Dados do Reservatório de Funil (RJ)](#dados-do-reservatório-de-funil-rj)
5. 🥴​ [Processamento e Tratamento dos Dados](#processamento-e-tratamento-dos-dados)
   - [Tratamento Individual dos Dados](#tratamento-individual-dos-dados)
   - [Unificação dos Dataframes](#unificação-dos-dataframes)
6. 🧐 [Análise Exploratória e Visualização](#análise-exploratória-e-visualização)
7. 🫡​ [Modelagem e Treinamento da Rede Neural](#modelagem-e-treinamento-da-rede-neural)
8. 🤔​ [Discussão dos Resultados](#discussão-dos-resultados)
9. 😉 [Conclusão e Próximos Passos](#conclusão-e-próximos-passos)
10. ​​😎​ [Estrutura do Projeto](#estrutura-do-projeto)
11. 🤩​ [Contribuição](#contribuição)
12. 🤓​ [Links e Referências](#links-e-referências)


## 😁​
## Introdução

A água potável é essencial para a sobrevivência humana e a gestão de seus recursos depende, em grande medida, do controle e monitoramento dos níveis dos reservatórios. Contudo, o represamento não garante que o volume disponível seja suficiente para suprir as demandas da população ou preservar a qualidade ambiental da região. Assim, a predição do volume útil — ou seja, a parte operacional do reservatório — torna-se um elemento crucial para o planejamento estratégico.

Neste contexto, o uso de técnicas de inteligência artificial, especialmente Redes Neurais Feed-Forward (FFNN), surge como uma abordagem promissora para identificar padrões em dados climáticos e hidrológicos, permitindo estimar com antecedência variações críticas do volume útil.


## 😀​
## Contextualização

Conforme destacado em diversas fontes (ex.: notícias do G1 e entrevistas com especialistas), o volume morto, que representa a porção de água armazenada abaixo das comportas das represas, não deve ser explorado para consumo devido a problemas de diluição e à possibilidade de contaminação. Por outro lado, a previsão do volume útil, por meio da análise de variáveis climáticas, possibilita melhores decisões operacionais e evita investimentos desnecessários – como a instalação de bombas para captação do volume morto.

Modelos de FFNN têm demonstrado eficiência em lidar com dados incompletos e variáveis de séries temporais, sendo bastante utilizados em estudos de previsão de chuva, enchentes e demais fenômenos meteorológicos.


## ​😏
## Descrição do Problema e Solução Proposta

O problema central envolve a complexidade da correlação entre as variáveis climáticas e o volume útil do reservatório de Funil. Dados ausentes, variáveis com diferentes escalas e a natureza não linear da relação entre esses fatores dificultam a modelagem preditiva.

**Solução Proposta:** Utilizar uma FFNN para:
- Receber como entrada as variáveis climáticas (evaporação, insolação, precipitação, temperatura, umidade e velocidade do vento) acrescidas da data convertida para Unix timestamp.
- Treinar o modelo para prever o volume útil, permitindo identificar padrões que auxiliem no planejamento antecipado e na gestão dos recursos hídricos.


## ​😅
## Coleta de Dados

A construção do modelo se baseia na integração de duas fontes principais:

### Dados da Estação Meteorológica de Resende (RJ)

- **Fonte:** INMET (BDMEP)  
- **Período:** 17/09/1983 a 18/03/2022  
- **Variáveis Coletadas:**
  - **Evaporação do Piche (mm)** → _evap_
  - **Insolação Total (h)** → _inso_
  - **Precipitação Total (mm)** → _prec_
  - **Temperatura Média Compensada (°C)** → _temp_
  - **Umidade Relativa do Ar (%)** → _umid_
  - **Velocidade Média do Vento (m/s)** → _vent_

A escolha desta estação se deu com base na proximidade (12 km a 15 km) do reservatório, com dados disponibilizados no formato CSV.

### Dados do Reservatório de Funil (RJ)

- **Fonte:** SAR da ANA  
- **Período:** 01/01/1993 a 05/12/2017  
- **Variáveis Coletadas:**
  - **Volume Útil (%)** → _volu_
  - **Data da Medição** → _data_

Os dados do reservatório foram coletados em conjunto com os dados meteorológicos para possibilitar a análise integrada.


## ​🥴
## Processamento e Tratamento dos Dados

### Tratamento Individual dos Dados

**Estação Meteorológica de Resende (RJ):**

- Leitura do arquivo CSV com o Pandas.
- Identificação de dados ausentes utilizando métodos como `isnull()` e cálculo da porcentagem de falhas; a coluna _vent_ apresentou 45% de dados faltantes.
- Renomeação das colunas para nomes simplificados (ex.: "VENTO, VELOCIDADE MEDIA DIARIA(m/s)" para _vent_).
- Remoção dos registros com dados ausentes (_dropna_) e redefinição do índice (_reset_index_).
- Conversão do tipo dos dados (ex.: datas convertidas para `datetime` e colunas numéricas ajustadas).

**Reservatório de Funil (RJ):**

- Leitura do arquivo CSV e renomeação da coluna de data para _data_.
- Seleção apenas das colunas necessárias e eliminação de colunas irrelevantes com o método `drop`.
- Conversão de dados, particularmente da coluna _volu_, garantindo a substituição correta de vírgulas por pontos.
- Neste conjunto, não houve dados faltantes.

### Unificação dos Dataframes

Após o tratamento individual, os dois conjuntos foram unidos utilizando o método `merge` do Pandas com base na coluna _data_:

```python
df_final = pd.merge(df_estacao, df_reservatorio, how='inner', on='data')
```

Esse procedimento resultou em um único dataframe com 5.891 registros finais, representando uma perda de cerca de 58% dos dados iniciais, mas garantindo a correspondência exata entre as datas.

Adicionalmente, para uso no modelo de machine learning, foi realizada a conversão dos dados de data para o formato Unix:

```python
df_final["DataUnix"] = df_final["data"].values.astype(np.int64) // 10**9
df_final.drop("data", axis=1, inplace=True)
df_final.rename(columns={"DataUnix": "data"}, inplace=True)
```


## ​🧐
## Análise Exploratória e Visualização

Realizou-se uma análise exploratória para identificar relações entre as variáveis. Por exemplo, gráficos comparativos entre a evaporação e a precipitação mostraram que, em 2015, houve uma queda acentuada no volume útil associada a altos níveis de evaporação, corroborando com notícias da época.

Um trecho do código utilizado para criar gráficos interativos com Plotly é apresentado a seguir:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Agrupamento e normalização dos dados por período mensal
df_mensal = df_final.groupby(pd.Grouper(key='data', freq='M')).mean().dropna()
series_data = df_mensal.index.astype('string')
series_data_presente = df_mensal[df_mensal['prec'] > 0].index.astype('string')
series_data_ausente = df_mensal[df_mensal['prec'] == 0].index.astype('string')
df_mensal_normal = (df_mensal - df_mensal.min()) / (df_mensal.max() - df_mensal.min())

# Criação do gráfico interativo
fig = make_subplots()
fig.add_trace(go.Bar(name='Presença', x=series_data_presente, y=df_mensal_normal.loc[df_mensal_normal['prec'] > 0, 'evap'],
                     marker=dict(color='#3d85c6')))
fig.add_trace(go.Bar(name='Ausência', x=series_data_ausente, y=df_mensal_normal.loc[df_mensal_normal['prec'] == 0, 'evap']))
fig.add_trace(go.Scatter(name='Volume Útil', x=series_data, y=df_mensal_normal['volu'], mode='lines', line=dict(color="#0000ff")))
fig.update_layout(title="Presença e Ausência de Chuva em Relação à Evaporação",
                  font=dict(family="Courier New, monospace", size=15, color="Purple"))
fig.show()
```


## ​🫡
## Modelagem e Treinamento da Rede Neural

O modelo de Rede Neural Feed-Forward (FFNN) foi implementado utilizando Python com TensorFlow/Keras e Scikit-learn. A seguir, os principais aspectos da modelagem:

**Arquitetura do Modelo**
    
- Camada de Entrada: 7 neurônios, representando as variáveis climáticas:

    - Evaporação (evap)
    - Insolação (inso)
    - Precipitação (prec)
    - Temperatura (temp)
    - Umidade (umid)
    - Velocidade do Vento (vent)
    - Data (em formato Unix)

- Camada Oculta: 1 camada com 666 neurônios e função de ativação ReLU.

- Camada de Saída: 1 neurônio para a previsão do volume útil (volu).

**Configurações de Treinamento**

- Divisão dos Dados:
    - Treino: 75%
    - Teste: 25% (Utiliza-se train_test_split do Scikit-learn com random_state = 42.)

- Normalização: 

    As variáveis de entrada (X) foram normalizadas via StandardScaler. As variáveis de saída (y) foram mantidas em seu formato original para representar valores reais.

- Parâmetros do Treinamento:
    - Otimizador: Nadam com learning_rate de 0.0003.
    - Função de Perda: Erro Quadrático Médio (MSE).
    - Early Stopping: Monitoramento de val_loss com paciência de 21 épocas, interrompendo o treinamento aos 694 ciclos dentro de um máximo de 2500 épocas.

O trecho abaixo resume a configuração e o treinamento:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping

# Configuração do EarlyStopping
parar_rede_neural = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=21)

# Construção do modelo
modelo = Sequential()
modelo.add(Dense(666, input_dim=7, activation='relu'))
modelo.add(Dense(1, activation='linear'))

# Compilação do modelo
modelo.compile(optimizer=Nadam(learning_rate=0.0003), loss='mse')

# Treinamento
historico = modelo.fit(X_treino, y_treino, epochs=2500, validation_data=(X_teste, y_teste),
                       callbacks=[parar_rede_neural], verbose=1)
```


## ​🤔
## Discussão dos Resultados

Após o treinamento, o modelo apresentou os seguintes resultados:

- **Eficiência:** O explained_variance_score obtido foi de aproximadamente 37%, indicando um ajuste não linear típico de problemas de regressão em séries temporais.

- **Erro Médio:** O mean_squared_error (após aplicação da raiz) apontou um erro médio de aproximadamente ±19.60% na previsão do volume útil.

- **Gráficos de Perdas:** Os gráficos demonstraram uma convergência suave das curvas de treinamento e teste, embora o desafio dos dados ausentes e a diversidade de unidades continuem impactando a precisão.

Várias tentativas foram realizadas para melhorar o desempenho (remoção ou inclusão da variável de data, ajustes no número de neurônios e camadas, modificação da taxa de aprendizado, etc.), mas a configuração apresentada se mostrou a mais robusta dentro das experimentações realizadas.


## 😉
## Conclusão e Próximos Passos

A aplicação de uma FFNN para a predição do volume útil do reservatório de Funil é um desafio que demanda cuidados na preparação dos dados e ajuste fino dos hiperparâmetros. Os resultados iniciais, com eficiência de 37% e erro médio de ±19.60%, indicam que, embora o modelo capte tendências relevantes, há espaço para melhorias.

Futuras Direções:

- **Otimização de Hiperparâmetros:** Utilizar ferramentas como o Keras Tuner para automatizar a busca por melhores configurações.

- **Técnica Fuzzy:** Explorar a normalização dos dados com métodos de lógica Fuzzy para remover a dependência das unidades.

- **Aprimoramento do Pré-Processamento:** Investigar alternativas para minimizar a perda de registros durante o tratamento de dados.

- **Experimentação com Modelos Alternativos:** Comparar a FFNN com outras abordagens de machine learning para robustecer as previsões.


## ​​😎
## Estrutura do Projeto

A organização dos arquivos segue uma estrutura principal, com apenas o conteúdo necessário para replicar esse projeto:

```
RedesNeuraisFFNN/
├── dados/
│   ├── Estacao_Meteorologica/
│   │   └── RESENDE_diaria.csv
│   └── Usina_Hidreletrica/
│       └── LFUNIL_diario.csv
├── notebooks/
│   └── Aprendizagem_de_máquina.ipynb
│   └── Engenharia_dos_dados.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

**Observação:** 

O arquivo `requirements.txt` é um documento utilizado em projetos Python para listar todas as bibliotecas e suas versões necessárias para executar o aplicativo. Ele facilita a instalação das dependências, permitindo que qualquer pessoa (ou servidor) possa replicar o ambiente de desenvolvimento ou execução usando o comando:

```Bash
pip install -r requirements.txt
```
- As versões apresentadas podem ser ajustadas conforme a necessidade ou compatibilidade do seu projeto.

- Se futuramente utilizar outros pacotes (por exemplo, se optar por pytorch em vez de tensorflow ou incluir algum pacote adicional para manipulação de dados), inclua-os neste arquivo para garantir que o ambiente fique configurado corretamente.


## 🤩​
## Contribuição

Contribuições para o aprimoramento do projeto são bem-vindas! Para colaborar:

1. Faça um fork do repositório.

2. Crie uma branch para sua feature:

    ```Bash
    git checkout -b feature/nova-feature
    ```

3. Realize os commits com suas alterações:

    ```Bash
    git commit -m "Adiciona nova feature"
    ```

4. Faça o push para sua branch:

    ```Bash
    git push origin feature/nova-feature
    ```

5. Abra um Pull Request para integração.


## 🤓​
## Links e Referências

- Repositório GitHub: https://github.com/MeusEstudos/RedesNeuraisFFNN

- Página do Projeto: https://meusestudos.github.io/RedesNeuraisFFNN/

---

_Licenciado sob a Licença MIT._

Esta documentação unificada apresenta uma visão completa do projeto, desde a definição do problema, coleta e tratamento de dados, análise exploratória, modelagem com rede neural, até a discussão dos resultados e encaminhamentos para o futuro. Sinta-se à vontade para ajustar ou ampliar qualquer seção conforme novas melhorias ou atualizações se façam necessárias.

---

**Principais Referências**

- A documentação deste README.md foi criada com o auxílio do DeepSeek e do Copilot. O DeepSeek não compartilha conversas; ele apenas extraiu o conteúdo dos arquivos do Jupyter Notebook que baixei. Em seguida, compilei o conteúdo gerado e o transformei em PDF para que o Copilot pudesse finalizá-lo, incorporando também o PDF do relatório técnico do meu TCC. Acesse [minha interação com o Copilot](https://copilot.microsoft.com/shares/kUTqRN7v8zM996i2R86tG).

- ANA – Agência Nacional de Águas. SAR – Sistema de Acompanhamento de Reservatórios.

- INMET – Banco de Dados Meteorológicos.

- G1, Folha de São Paulo e demais fontes jornalísticas relacionadas aos níveis dos reservatórios.

- NAGASELVI & DEEPA (2015): Artigo sobre previsão meteorológica utilizando FFNN e técnicas Fuzzy.

- OLIVEIRA, S. (2019). Titanic Passo a Passo com 8 Modelos ML Pt-br (código e análise no Kaggle).

- Documentação oficial do TensorFlow, Keras, Scikit-learn, Pandas, Plotly, Matplotlib e Seaborn.
