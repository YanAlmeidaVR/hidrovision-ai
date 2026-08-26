# HidroVision AI

Sistema de monitoramento e alerta antecipado de enchentes urbanas por visão
computacional e aprendizado de máquina.

**Equipe 0010 — 45ª FETIN 2026 · INATEL · Santa Rita do Sapucaí, MG**

---

## O problema

Em **fevereiro de 2026**, chuvas extremas atingiram a Zona da Mata Mineira.
Juiz de Fora e Ubá registraram **73 mortes** e mais de **5.500 desalojados**.
O acumulado de fevereiro na cidade chegou a 752 mm, mais de quatro vezes a
média histórica do mês, e dois dias daquela semana entraram entre os cinco mais
chuvosos já medidos desde 1961.

Dois anos antes, em **maio de 2024**, o Rio Grande do Sul viveu a maior
catástrofe climática de sua história: **183 mortes**, 478 municípios atingidos,
2,4 milhões de pessoas afetadas e 442 mil obrigadas a deixar suas casas.

O que chama atenção no caso gaúcho é que **o aviso existia**. Entre 26 de abril
e 5 de maio de 2024, o INMET emitiu 26 alertas de tempo severo para o estado,
seis deles vermelhos, de grande perigo. A informação meteorológica estava
disponível. O que faltou foi traduzi-la, no nível de cada município e de cada
bairro, em uma resposta concreta: *quanto tempo ainda temos até a água chegar
aqui?*

Essa lacuna tem uma causa prática. Monitorar o nível de um rio exige
equipamento, e sistemas convencionais custam dezenas de milhares de reais por
ponto de medição — inviável para a maior parte dos mais de 5.500 municípios
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
  hidrovision_v05.pt    modelo YOLO26n treinado (13 classes)

preditivo/              módulo de previsão
  DadosANA.py           baixa o nível da ANA (API autenticada)
  MergeInmet.py         lê os pacotes do INMET e junta as séries
  grafico_nivel.py      visualização da série histórica
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
  apresentacao.py       modo de demonstração
  pipeline.py           orquestrador

docs/                   relatórios técnicos e figuras
```

---

## Módulo de visão computacional

Detector **YOLO26n** (2,38 M parâmetros) treinado para localizar a régua e os
números gravados nela. Treze classes: os onze números de 0 a 100 (de 10 em 10),
a régua (`gauge`) e a linha d'água (`surface`).

### Resultados (conjunto de teste, 608 imagens)

| Métrica | Valor |
|---|---|
| mAP@50 global | 90,5% |
| mAP@50 das 12 classes úteis | 93,2% |
| Recall das classes numéricas | 0,82 a 0,95 (todas acima da meta de 0,80) |
| Recall da régua | 0,984 |
| Confusão entre números | ≤ 1% |

O erro dominante é a **não detecção** (≈ 11,5%), não a classificação
incorreta: o modelo raramente troca um número por outro.

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
python visao/mdYOLO.py --modelo visao/hidrovision_v05.pt --imagem foto.jpg
python visao/mdYOLO.py --modelo visao/hidrovision_v05.pt --webcam 0
python visao/mdYOLO.py --modelo visao/hidrovision_v05.pt --pasta ./fotos --csv leituras.csv
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

| Horizonte | MAE do modelo | MAE da persistência | Ganho | Em subidas ≥ 20 cm |
|---|---|---|---|---|
| t+6h | 1,27 cm | 2,68 cm | +53% | +72% |
| t+12h | 2,44 cm | 5,16 cm | +53% | +73% |
| t+24h | 4,74 cm | 9,62 cm | +51% | +69% |

A última coluna é a que importa: prever um rio parado é trivial, e a média
global é dominada por essas horas. Nas subidas fortes — os eventos que
justificam o sistema — o ganho chega a 72%.

Na detecção do cruzamento dos limiares de alerta (228, 304 e 388 cm, percentis
90, 95 e 99 do histórico da estação), o recall ficou entre 0,95 e 0,98 em t+6h
sobre 63 horas de emergência real ocorridas em 2026.

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

---

## Limitações reconhecidas

- **A classe `surface` não convergiu** no conjunto de teste (recall 0,388). A
  linha d'água não tem forma consistente: varia com reflexo, turbidez e sombra.
  A leitura se apoia na geometria dos números.
- **O erro de leitura em centímetros ainda não foi medido em campo.** A
  resolução garantida é a faixa de 10 cm.
- **A chuva é fator secundário nos horizontes curtos.** O nível recente já
  carrega o efeito da chuva que caiu; o modelo é de inércia hidrológica, não um
  modelo chuva-vazão completo.
- **Os modelos foram treinados com chuva observada, não com previsões.**
  Incorporar a chuva prevista às variáveis de acumulado é uma aproximação
  válida, já que a física do escoamento não distingue a origem do dado, mas
  herda o erro da previsão meteorológica.
- **O horizonte máximo é de 24 horas.** Antecipações mais longas vêm da previsão
  meteorológica, não dos modelos.

---

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
