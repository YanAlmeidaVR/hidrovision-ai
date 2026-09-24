# HidroVision AI

Sistema de monitoramento e alerta antecipado de enchentes urbanas por visão
computacional e aprendizado de máquina.

**Equipe 0010 — 45ª FETIN 2026 · INATEL · Santa Rita do Sapucaí, MG**

---

## O problema

Em **fevereiro de 2026**, chuvas extremas atingiram a Zona da Mata Mineira.
Juiz de Fora e Ubá registraram **73 mortes** e mais de **5.500 desalojados**.
O acumulado de fevereiro na cidade chegou a 752 mm, quatro vezes e meia a
média histórica do mês, e dois dias daquela semana entraram entre os cinco mais
chuvosos já medidos desde 1961.

Dois anos antes, em **maio de 2024**, o Rio Grande do Sul viveu a maior
catástrofe climática de sua história: **184 mortes**, 478 municípios atingidos,
2,4 milhões de pessoas afetadas e 442 mil obrigadas a deixar suas casas.

O que chama atenção no caso gaúcho é que **o aviso existia**. Entre 26 de abril
e 5 de maio de 2024, o INMET emitiu 26 alertas de tempo severo para o estado,
seis deles vermelhos, de grande perigo. A informação meteorológica estava
disponível. O que faltou foi traduzi-la, no nível de cada município e de cada
bairro, em uma resposta concreta: *quanto tempo ainda temos até a água chegar
aqui?*

Essa lacuna tem uma causa prática. Monitorar o nível de um rio exige
equipamento, e sistemas convencionais custam dezenas de milhares de reais por
ponto de medição, inviável para a maior parte dos mais de 5.500 municípios
brasileiros. Onde não há sensor, a leitura é manual e o alerta chega tarde, ou
simplesmente não chega.

## A proposta

Uma câmera de baixo custo aponta para uma régua linimétrica instalada **junto à
área urbana**, não no leito do rio. Nessa configuração, os 100 cm da régua
correspondem ao ponto em que a água atinge a cidade: a régua funciona como uma
contagem regressiva, e não como um medidor hidrológico. Água na marca de 10 cm
significa que ainda restam 90 cm de folga.

O sistema opera em duas camadas que se confirmam:

| Camada | Fonte | Papel |
|---|---|---|
| **Previsão** | dados da ANA e previsão meteorológica | antecipa: o rio deve subir X cm nas próximas horas |
| **Observação** | câmera lendo a régua | confirma: a água chegou e continua subindo |

Quando o modelo indica elevação e a régua confirma a subida, a previsão deixa
de ser hipótese e passa a ser fato observado.

---

## Estrutura do repositório

```
visao/                  módulo de visão computacional
  mdYOLO.py             leitura do nível a partir da imagem
  geometria.py          validação e correção geométrica das detecções
  servidor_camera.py    servidor de câmera para a Raspberry Pi (modo rede)
  hidrovision_v06_regua.pt            modelo YOLO26n com fine-tuning na régua própria
  hidrovision_v06_regua_w8a32.tflite  versão quantizada, embarcada no Raspberry Pi
  hidrovision_v05.pt    modelo anterior, treinado na régua de referência
  HidroVision_FineTuning_Regua.ipynb  notebook do fine-tuning v05 → v06

preditivo/              módulo de previsão
  DadosANA.py           baixa o nível da ANA (API autenticada)
  MergeInmet.py         lê os pacotes do INMET e junta as séries
  grafico_nivel.py      visualização da série histórica
  otimizar.py           otimização de hiperparâmetros (Optuna)
  dados_treino.csv      dataset final: 27.005 horas de dados reais
  HidroVision_XGBoost.ipynb   treino dos modelos
  modelos/              modelos XGBoost treinados + metadados

integracao/             camada que liga leitura, previsão e alerta
  banco.py              SQLite: leituras, previsões e alertas
  tendencia.py          velocidade de variação do nível, em cm/h
  projecao.py           tempo restante até a água atingir a área urbana
  alertas.py            alertas de nível e de trajetória, com histerese
  preditor.py           carrega os modelos e monta as variáveis
  clima.py              previsão de chuva (Open-Meteo)
  monitor.py            ciclo horário de monitoramento
  pipeline.py           orquestrador
  configurar_telegram.py  descobre o chat_id e testa o envio de alertas
  demo_simulada.py      demonstração sem câmera (maquete simulada ou replay da ANA)
  demo_maquete.py       demonstração com a maquete física
  dashboard.py          painel Streamlit: estação (monitoramento e simulação) e régua urbana
  regua_ao_vivo.py      painel da régua urbana: câmera ao vivo e previsão

docs/                   relatórios técnicos e figuras
```

---

## Módulo de visão computacional

Detector **YOLO26n** (2,38 M parâmetros) treinado para localizar a régua e os
números gravados nela. Treze classes: os onze números de 0 a 100 (de 10 em 10),
a régua (`gauge`) e a linha d'água (`surface`).

### Dois conjuntos de teste, duas perguntas diferentes

O modelo passou por duas etapas de treino, e cada uma responde a uma pergunta
distinta. Os números não são comparáveis entre si e estão separados de
propósito.

**Etapa 1 — o modelo genérico (V05), sobre réguas de vários tipos**
*Conjunto de teste: 608 imagens do dataset público.*

| Métrica | Valor |
|---|---|
| mAP@50 global | 90,5% |
| mAP@50 das 12 classes úteis | 93,2% |
| Recall das classes numéricas | 0,82 a 0,95 |
| Recall da régua | 0,984 |
| Confusão entre números | ≤ 1% |

O erro dominante é a **não detecção** (≈ 11,5%), não a classificação
incorreta: o modelo raramente troca um número por outro.

**Etapa 2 — o modelo em operação (V06), sobre a régua fabricada**
*Conjunto de teste: imagens da régua da maquete, em dois cenários de
iluminação e enquadramento.*

| Modelo | mAP@50 na régua fabricada |
|---|---|
| V05, sem fine-tuning | **0,109** |
| V06, após fine-tuning (382 imagens) | **0,840** |

Esse salto é o resultado mais importante do módulo. Um detector treinado em
réguas genéricas praticamente não enxerga uma régua específica que nunca viu:
0,109 significa que o sistema não funcionaria em campo. O fine-tuning com
imagens da própria régua — congelando as 10 primeiras camadas, `lr0=0.001`,
AdamW — resolve o problema com algumas centenas de fotos, o que é reproduzível
por qualquer prefeitura que instale a sua própria régua.

**O V06 é o modelo que roda na demonstração.** O valor de referência é
**0,840**.

### Modelo quantizado para o embarcado

O V06 foi exportado para LiteRT com quantização de pesos em 8 bits (`w8a32`),
caindo de 9,4 MB para **2,7 MB**. Numa validação lado a lado, sobre um recorte
de teste construído separadamente, o `.pt` marcou 0,868 e o `.tflite` marcou
0,873 — ou seja, **a quantização não produziu perda mensurável**. Como esse
recorte não é o mesmo da tabela acima, esses dois valores servem apenas para
comparar os formatos entre si; o número oficial do modelo continua sendo 0,840.

### Da detecção à leitura

As marcações da régua são monotônicas e igualmente espaçadas, o que permite
determinar em que faixa está a lâmina de água a partir de quais números
permanecem visíveis. A mesma restrição geométrica é usada para validar as
detecções entre si, corrigindo classificações incoerentes de baixa confiança.

Com dois ou mais números visíveis a leitura é refinada; com apenas um, a
resolução é a faixa de 10 cm. A interface informa a faixa, sem apresentar
precisão que ainda não foi medida em campo.

### Uso

```bash
python visao/mdYOLO.py --modelo visao/hidrovision_v07_regua.pt --imagem foto.jpg
python visao/mdYOLO.py --modelo visao/hidrovision_v07_regua.pt --webcam 0
python visao/mdYOLO.py --modelo visao/hidrovision_v07_regua.pt --pasta ./fotos --csv leituras.csv
```

---

## Módulo preditivo

Modelos **XGBoost** que estimam a cota futura do Rio Sapucaí, construídos
inteiramente sobre dados reais de fontes oficiais.

### Dados

| Fonte | Dado | Estação |
|---|---|---|
| ANA — HidroWebService | nível do rio, a cada 15 min | 61305000 — Santa Rita do Sapucaí |
| INMET — histórico | chuva horária | A531 — Maria da Fé (bacia alta) |

**Dataset final:** 27.005 horas (jan/2023 a ago/2026), cobertura de 94,3% no
nível e 100% na chuva, nível variando de 14 a 447 cm.

A escolha da estação meteorológica foi hidrológica: Maria da Fé está a montante,
na Serra da Mantiqueira, e mede a chuva que ainda vai escoar até o trecho
monitorado.

A junção das duas séries exigiu decisões específicas: agregação horária (média
do nível, soma da chuva), conversão de UTC para hora local, tratamento do valor
−9999 do INMET como ausência e não como zero, reindexação em grade horária
contínua para que as defasagens não atravessem lacunas de telemetria, e
preenchimento das falhas do INMET com o pluviômetro da própria estação da ANA.

### Os modelos

Três modelos preveem a **variação** do nível (não o valor absoluto) em t+6h,
t+12h e t+24h.

Essa escolha foi determinante. Prevendo o nível absoluto, a resposta está quase
inteiramente contida no nível atual — o modelo aprende a copiá-lo e a chuva fica
com 0,2% de importância. Prevendo a variação, a chuva se torna informativa:

| Alvo | t+6h | t+12h | t+24h |
|---|---|---|---|
| nível absoluto | 0,2% | 0,7% | 3,4% |
| **variação** | **14,2%** | **20,6%** | **41,1%** |

E o erro melhorou junto: não houve troca de precisão por interpretabilidade.

### Desempenho

Validação temporal: treino de 2023 a 2025, teste em 2026. O critério é superar o
**baseline de persistência** (prever que o nível não muda), exigente porque em
regime de estiagem ele acerta na maior parte das horas.

| Horizonte | MAE do modelo | MAE da persistência | Ganho |
|---|---|---|---|
| t+6h | 1,26 cm | 2,68 cm | +53% |
| t+12h | 2,60 cm | 5,16 cm | +50% |
| t+24h | 5,47 cm | 9,62 cm | +43% |

O erro cresce com o horizonte, como esperado, mas a vantagem sobre a
persistência se mantém acima de 40% nos três casos. Prever um rio parado é
trivial, e a média global é dominada por essas horas — o valor do modelo
aparece nas horas em que o rio se move.

Na detecção do cruzamento dos limiares de alerta (228, 304 e 388 cm, percentis
90, 95 e 99 do histórico da estação), o recall ficou entre 0,95 e 0,98 em t+6h
sobre 63 horas de emergência real ocorridas em 2026.

Os valores acima são os gravados em `preditivo/modelos/metricas_delta.csv` e
correspondem aos modelos versionados em `preditivo/modelos/`.

### Validação em evento real — setembro de 2026

Entre 10 e 12 de setembro de 2026 o Rio Sapucaí subiu de 87 cm para 305 cm e
provocou inundação urbana. O monitor estava em operação contínua e registrou o
evento do começo ao fim.

| | |
|---|---|
| Antecedência do primeiro alerta | **66 horas** |
| Nível no primeiro alerta | 87 cm |
| Pico registrado | 305 cm |
| Erro médio no horizonte de 6 h durante o evento | **1,3 cm** |

O primeiro alerta de nível saiu às 08:55 do dia 10, com o rio ainda a 87 cm, e
os alertas seguintes acompanharam a subida ao longo dos dois dias. Nos
horizontes mais longos as projeções **subestimaram** o pico — às 08:55 o modelo
projetava 173–175 cm onde o rio chegou a 194 cm, e o padrão se repetiu nos
alertas posteriores. A direção e o cruzamento dos limiares foram detectados
corretamente; a magnitude do pico, não. Para a finalidade do sistema — avisar
com antecedência que o rio vai subir e atingir a cidade — o comportamento foi
adequado, mas a subestimação em 12 h e 24 h é uma limitação conhecida e
registrada.

O estudo de caso completo, com a série do nível, os alertas emitidos e as
capturas do Telegram, está em `docs/caso_setembro.html`.

### Uso

```bash
# baixar os dados (requer cadastro na ANA)
export ANA_ID=seu_cpf ANA_SENHA=sua_senha
python preditivo/DadosANA.py --estacao 61305000 --inicio 2023-01-01 --fim 2026-08-09

# juntar com a chuva do INMET (baixe os ZIPs anuais do portal antes)
python preditivo/MergeInmet.py --estacao-inmet A531

# treinar: abra o notebook no Colab e envie o dados_treino.csv
```

Para usar um modelo já treinado, lembre que ele prevê a variação:

```python
import xgboost as xgb
m = xgb.XGBRegressor()
m.load_model("preditivo/modelos/modelo_delta_6h.json")
nivel_previsto = nivel_atual + m.predict(features)[0]
```

A ordem exata das 23 variáveis está em `preditivo/modelos/features_delta.json`.

---

## Camada de integração

Transforma leitura em decisão. Cada leitura é gravada em SQLite com filtro de
mediana, para que uma detecção espúria não contamine o histórico. A partir da
série, o sistema calcula a velocidade de subida e projeta quanto tempo falta
até a água atingir os 100 cm.

Os alertas são de **duas categorias independentes**:

| Categoria | O que é | Ciclo de vida |
|---|---|---|
| **Nível** | a água está a X cm — um fato, vale em qualquer direção | dispara ao cruzar o limiar; não desarma ao começar a descer; normaliza abaixo de (limiar − folga) |
| **Trajetória** | subindo, atinge os 100 cm em X horas — uma previsão | dispara pela faixa de tempo restante; cancela quando a subida cessa |

A separação importa: água a 30 cm subindo 40 cm/h atinge a área urbana em menos
de duas horas, enquanto água a 90 cm parada não tem trajetória alguma. Um alerta
único não distinguiria os dois casos, e o primeiro é o que exige ação imediata.

### Operação contínua

```bash
export ANA_ID=seu_cpf ANA_SENHA=sua_senha
python integracao/monitor.py --modelos preditivo/modelos --ciclos 0 --intervalo 60
```

A cada hora o monitor consulta o nível corrente na estação da ANA, busca a
previsão de chuva e executa os modelos em **dois cenários** — com e sem a chuva
prevista. A diferença entre eles quantifica o impacto esperado da precipitação.


## Instalação

```bash
git clone https://github.com/YanAlmeidaVR/hidrovision-ai.git
cd hidrovision-ai
pip install -r requirements.txt
```

## Documentação

Relatórios técnicos completos em `docs/`.

## Tecnologias

Python · Ultralytics YOLO26 · XGBoost · OpenCV · pandas · SQLite · Roboflow ·
Open-Meteo · Kaggle · Google Colab

## Fontes dos dados citados

- Enchentes na Zona da Mata Mineira, fevereiro de 2026 — Agência Brasil, Agência
  Pública e INMET
- Enchentes no Rio Grande do Sul, maio de 2024 — Defesa Civil do RS,
  Confederação Nacional de Municípios e INMET
- Série histórica de nível — Agência Nacional de Águas, estação 61305000
- Série histórica de precipitação — INMET, estação automática A531
