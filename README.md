# HidroVision AI

Sistema de monitoramento e alerta antecipado de enchentes urbanas por visão
computacional e aprendizado de máquina.

**Equipe 0010 — 45ª FETIN 2026 · INATEL · Santa Rita do Sapucaí, MG**

---

## O problema

Enchentes urbanas causam prejuízos bilionários no Brasil, e a maioria dos
municípios de pequeno e médio porte não tem sistema de alerta. Os métodos
existentes dependem de leitura manual da régua ou de sensores caros, sem
integração preditiva — o resultado são alertas tardios ou inexistentes.

## A proposta

Uma câmera de baixo custo aponta para uma régua linimétrica instalada **junto à
área urbana**, não no leito do rio. Nessa configuração, os 100 cm da régua
correspondem ao ponto em que a água atinge a cidade: a régua funciona como uma
contagem regressiva, e não como um medidor hidrológico.

O sistema tem dois módulos independentes:

| Módulo | O que faz |
|---|---|
| **Visão computacional** | detecta a régua e os números, determinando em que faixa está a lâmina de água |
| **Preditivo** | prevê a cota do rio na bacia a partir do histórico de nível e chuva |

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
  HidroVision_Horizontes_Longos.ipynb   treino dos modelos
  modelos/              modelos XGBoost treinados + metadados

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

O erro dominante é a **não detecção** (≈ 11,5%), não a classificação incorreta:
o modelo raramente troca um número por outro.

### Da detecção à leitura

As marcações da régua são monotônicas e igualmente espaçadas (10 cm entre
números), então a própria régua fornece a escala: se o 20 está 50 pixels acima
do 10, então 1 cm equivale a 5 pixels. Nenhuma calibração manual é necessária.

O `geometria.py` usa essa restrição para **corrigir erros de classificação**.
Um ajuste robusto por consenso determina a reta que relaciona valor e posição;
detecções incoerentes com essa reta e de baixa confiança são reclassificadas.

> Em ensaio real, uma sequência detectada como 90 · 80 · 70 · 60 · 60 teve a
> última caixa corrigida para 50 — a leitura passou de 60 cm (errada) para
> 48 cm (correta).

**Resolução:** com dois ou mais números visíveis, a interpolação atinge cerca de
2 a 3 cm. Com apenas um número, a leitura é o próprio valor dele — degraus de
10 cm. A interface informa a faixa, sem apresentar precisão não medida em campo.

### Uso

```bash
python visao/mdYOLO.py --modelo visao/hidrovision_v05.pt --imagem foto.jpg
python visao/mdYOLO.py --modelo visao/hidrovision_v05.pt --webcam 0
python visao/mdYOLO.py --modelo visao/hidrovision_v05.pt --pasta ./fotos --csv leituras.csv
```

---

## Módulo preditivo

Modelos **XGBoost** que estimam a cota futura do Rio Sapucaí. Construídos
inteiramente sobre dados reais de fontes oficiais.

### Dados

| Fonte | Dado | Estação |
|---|---|---|
| ANA — HidroWebService | nível do rio, a cada 15 min | 61305000 — Santa Rita do Sapucaí |
| INMET — histórico | chuva horária | A531 — Maria da Fé (bacia alta) |

**Dataset final:** 27.005 horas (jan/2023 a ago/2026), cobertura de 94,3% no
nível e 100% na chuva, nível variando de 14 a 447 cm.

A junção das duas séries exigiu decisões específicas: agregação horária (média
do nível, soma da chuva), conversão de UTC para hora local, tratamento do valor
−9999 do INMET como ausência e não como zero, reindexação em grade horária
contínua para que as defasagens não atravessem lacunas de telemetria, e
preenchimento das falhas do INMET com o pluviômetro da própria estação da ANA.

### Modelos

Três modelos que preveem a **variação** do nível (não o valor absoluto) em
t+6h, t+12h e t+24h.

Essa escolha foi determinante. Prevendo o nível absoluto, a resposta está quase
inteiramente contida no nível atual — o modelo aprende a copiá-lo e a chuva fica
com 0,1% de importância. Prevendo a variação, a chuva se torna informativa:

| Alvo | Peso da chuva em t+6h | t+12h | t+24h |
|---|---|---|---|
| nível absoluto | 0,2% | 0,7% | 3,4% |
| **variação** | **14,2%** | **20,6%** | **41,1%** |

E o erro melhorou junto — não houve troca de precisão por interpretabilidade.

### Desempenho

Validação temporal: treino de 2023 a 2025, teste em 2026. O critério é superar o
**baseline de persistência** (prever que o nível não muda).

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

A ordem exata das 23 features está em `preditivo/modelos/features_delta.json`.

## Instalação

```bash
git clone https://github.com/<usuario>/hidrovision-ai.git
cd hidrovision-ai
pip install -r requirements.txt
```

## Documentação

Os relatórios técnicos completos estão em `docs/relatorios/`, cobrindo o modelo
de visão, o módulo preditivo e a arquitetura de integração.

## Tecnologias

Python · Ultralytics YOLO26 · XGBoost · OpenCV · pandas · SQLite · Roboflow ·
Kaggle · Google Colab
