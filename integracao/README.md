# HidroVision AI — Fase 3 (Integração)

Camada que liga a leitura da câmera aos modelos preditivos e aos alertas.

```
leitura (YOLO) ──> banco.py ──> serie horária ──> preditor.py
                      │                                │
                 tendencia.py                    previsões 6/12/24h
                      │                                │
                      └────────> alertas.py <──────────┘
                                    │
                        console / log / (Telegram depois)
```

## Arquivos

| Arquivo | Papel |
|---|---|
| `banco.py` | SQLite: leituras, alertas, série horária, importação do CSV da ANA |
| `tendencia.py` | cm/h; detecta a resolução da leitura e troca de método |
| `alertas.py` | dois alertas independentes (nível e trajetória) com histerese |
| `projecao.py` | tempo até a água atingir os 100 cm (área urbana) |
| `preditor.py` | carrega os `modelo_delta_*.json` e prevê; inclui simulação de chuva |
| `pipeline.py` | orquestrador: uma chamada por leitura faz o ciclo inteiro |
| `clima.py` | previsão de chuva pela API do Open-Meteo (sem chave) |
| `monitor.py` | laço contínuo: consulta a ANA + previsão, prevê e classifica o risco |
| `demo_simulada.py` | demonstração sem câmera (maquete simulada ou replay da ANA) |
| `apresentacao.py` | modo de feira: rio real e régua urbana lado a lado |
| `cenario_chuva.json` | previsão congelada, para não depender de rede no dia |

Os `modelo_delta_6h/12h/24h.json` precisam estar na pasta indicada em
`pasta_modelos` (padrão: a atual).

## Uso no loop da câmera (Raspberry Pi)

```python
from pipeline import Pipeline

p = Pipeline(modo="maquete", pasta_modelos="modelosPreditivos")

# a cada leitura do YOLO:
r = p.processar_leitura(nivel_cm=48.2, metodo="geometria",
                        confianca=0.71, menor_numero=50)
# r.nivel_gravado  -> valor após o filtro de mediana
# r.tendencia      -> taxa em cm/h + rótulo (subindo/estável/descendo)
# r.previsoes      -> {'nivel_atual':.., '6h':.., '12h':.., '24h':..}
# r.eventos_alerta -> disparos/rearmes gerados nesta leitura
```

## Uso pelo dashboard

```python
p.estado_atual()          # card de status: nível, tendência, previsões, limiares
p.simular_chuva(10)       # slider: previsões com chuva hipotética de 10 mm/h
p.trocar_modo("estacao")  # alterna limiares maquete <-> estação real
p.banco.serie_recente(48) # gráfico temporal
p.banco.alertas_recentes()# tabela de alertas
```

## Monitoramento contínuo

O `monitor.py` fecha o ciclo de operação. A cada intervalo (padrão: 1 hora):

1. consulta o **nível atual** do rio na estação 61305000 (API da ANA);
2. consulta a **previsão de chuva** para a bacia alta (Open-Meteo, sem chave);
3. grava no banco;
4. roda os modelos em **dois cenários** — sem chuva adicional e com a chuva
   prevista — o que mostra explicitamente quanto a previsão agrava o quadro;
5. combina com o estado da régua urbana e classifica o risco.

```bash
export ANA_ID=seu_cpf ANA_SENHA=sua_senha
python monitor.py --modelos ../modelos --importar dados_treino.csv --ciclos 1
python monitor.py --modelos ../modelos --intervalo 60 --ciclos 0   # contínuo
```

### As duas camadas

| Camada | Fonte | Papel |
|---|---|---|
| **Rio** | estação da ANA + previsão de chuva | antecipa: "o rio deve subir 80 cm em 6 h" |
| **Régua urbana** | câmera | confirma: a água chegou à régua e continua subindo |

A régua de 1 m fica junto à área urbana: os 100 cm são o ponto em que a água
atinge a cidade. Água no 10 significa 90 cm de folga — é o começo do
monitoramento, não o transbordamento.

Quando as duas camadas concordam — modelo prevendo elevação **e** régua
confirmando a subida — o risco é escalado e a mensagem registra a confirmação
cruzada. É o caso em que a previsão deixa de ser hipótese e passa a ser
observação.

## Operação contínua

```bash
export ANA_ID=seu_cpf ANA_SENHA=sua_senha
python monitor.py --modelos ../preditivo/modelos --ciclos 0 --intervalo 60
```

Com `--ciclos 0` o monitor fica em laço permanente. Cada ciclo grava na tabela
`previsoes`: nível corrente, previsões para 6/12/24 h nos dois cenários, o
resumo meteorológico e o nível de risco. É desse histórico que o dashboard
monta os gráficos.

A separação é proposital: o monitor coleta e decide, o dashboard apenas exibe.
Colocar a coleta dentro da interface faria a série parar sempre que ninguém
estivesse com a página aberta.

O servidor da ANA devolve erros 5xx com alguma frequência, então a consulta é
repetida três vezes com espera crescente antes de desistir. Se ainda assim
falhar, o ciclo segue com o que já está no banco e registra o aviso.

## Modo de apresentação

O `apresentacao.py` foi feito para a feira. Ele exibe as duas camadas ao mesmo
tempo: o rio subindo, reproduzido a partir de um evento **real** registrado pela
ANA em março de 2026, e a régua urbana da maquete, lida pela câmera.

A previsão de chuva vem de `cenario_chuva.json`, um arquivo preparado antes.
Numa feira, depender de rede é risco desnecessário; o arquivo descreve o
cenário narrado ("128 mm nos próximos 3 dias") e o sistema funciona offline.

```bash
# ensaio, sem câmera (a régua também é simulada)
python apresentacao.py --dados dados_treino.csv --modelos ../preditivo/modelos

# no dia, com a câmera lendo a maquete
python apresentacao.py --dados dados_treino.csv --modelos ../preditivo/modelos --camera

# velocidade: segundos de exibição por hora de dados
python apresentacao.py --seg-por-hora 0.5
```

A tela mostra, a cada hora simulada, o nível do rio com a previsão para 24 h em
dois cenários, o nível da régua com a folga restante, e o risco combinado.

Uma observação sobre o horizonte: **os modelos preveem no máximo 24 horas**. A
antecipação de vários dias vem da previsão meteorológica, não deles. A narrativa
correta é que a previsão indica chuva forte nos próximos dias e o modelo traduz
isso em elevação do rio dentro do horizonte que domina.

## Demonstrações

```bash
python demo_simulada.py                    # enchente na maquete (simulada)
python demo_simulada.py --modo estacao --replay-ana dados_treino.csv
```

```bash
python demo_simulada.py --interpolada      # leitura fina, se a geometria ajudar
```

**A demo da maquete simula leitura em degraus de 10 cm**, que é o que a câmera
realmente produz: ela informa o menor número visível na régua, e esse valor só
muda quando a água cobre a marcação seguinte. A coluna "real" mostra o nível
verdadeiro (que só a água sabe) apenas para comparação.

O replay reproduz a cheia real de março/2026 com os dados da estação. Resultado
registrado: o alerta antecipado (previsão 6h) disparou às 04:00 de 12/03; o
nível só cruzou o limiar às 06:00 — **2 h de antecedência em evento real**.

## O que a régua representa

A régua de 1 m **não fica no rio medindo cota** — ela fica junto à área urbana.
Os 100 cm são o ponto em que a água atinge a cidade; quanto mais alto o número
visível, menor a folga restante. A régua é, portanto, uma contagem regressiva.

Disso decorre a lógica de alerta, dividida em **duas categorias independentes**:

| Categoria | O que é | Ciclo de vida |
|---|---|---|
| **NÍVEL** | a água está a X cm — um fato, vale em qualquer direção | dispara ao cruzar o limiar; **não** desarma ao começar a descer; normaliza abaixo de (limiar − folga) |
| **TRAJETÓRIA** | subindo, atinge os 100 cm em X horas — uma previsão | dispara pela faixa de tempo restante; **cancela** quando a subida cessa |

A separação importa: água a 30 cm subindo 40 cm/h (crítico em 1h45) é mais
urgente que água a 90 cm parada. Um alerta único não distinguiria os casos.

Faixas de urgência da trajetória: menos de 6 h → atenção, menos de 3 h →
alerta, menos de 1 h → emergência.

A projeção é uma extrapolação explicável (folga ÷ taxa), não um modelo
estatístico — e usa apenas a resolução que a régua oferece.

## Limitação importante: os modelos XGBoost não valem na régua urbana

Os modelos XGBoost aprenderam a dinâmica do Rio Sapucaí na estação 61305000,
onde o nível varia de 14 a 447 cm e o rio responde em horas. A maquete opera em
0-100 cm e enche em minutos — outra escala e outra física. Alimentar os modelos
com dados da maquete produz previsão sem sentido (chegou a prever estiagem
enquanto a água subia 18 cm/h).

Por isso a régua urbana usa a projeção por tendência, e os modelos XGBoost
ficam para os dados da estação. A divisão é:

| Demonstração | O que prova |
|---|---|
| maquete (câmera + água) | leitura da régua, tendência, projeção até os 100 cm, alertas |
| replay dos dados da ANA | previsão e alerta antecipado em cheia real |

Treinar um preditor para a maquete exigiria registrar enchimentos reais dela —
possível, mas é outro dataset e outro modelo.

## Decisões de projeto embutidas

- **Tendência adaptativa à resolução**: se a leitura é interpolada (~2-3 cm),
  usa regressão em 30 min. Se vem em degraus de 10 cm (só um número visível),
  mede o tempo entre cruzamentos de degrau em 2 h — regressão sobre a escada
  daria valores oscilando entre 0 e o dobro do real. Sem cruzamento na janela,
  reporta "estável" com o limite superior (|taxa| < passo/janela) em vez de
  zero.
- **Mediana antes de gravar**: outlier de detecção não entra no histórico
  (e portanto não contamina lags nem tendência). Efeito colateral: em subida
  muito rápida a gravação atrasa ~2 leituras — aceitável.
- **Agregação horária**: os modelos foram treinados com dado horário; o
  `serie_horaria()` faz a média por hora antes de montar features.
- **Histerese**: dispara ao cruzar o limiar, só rearma abaixo de
  (limiar − folga). Folga: 5 cm na maquete, 15 cm na estação.
- **Alerta antecipado**: a previsão também é comparada aos limiares — é isso
  que gera o aviso antes de o nível cruzar.
- **Previsão a cada N min** (padrão 10), não a cada frame: o XGBoost é leve,
  mas não precisa rodar 2× por minuto no Pi.
- **Canais plugáveis**: hoje console + arquivo. O Telegram entra criando uma
  classe `CanalTelegram` com o método `enviar(mensagem)` — o esqueleto já
  está comentado em `alertas.py`.

## Pendências desta fase

- Plugar o `CanalTelegram` (token do BotFather + chat_id).
- Conectar o `leitura_nivel.py` real: no modo webcam, chamar
  `p.processar_leitura(...)` no lugar do print.
- Chuva em tempo real: hoje a simulação injeta chuva manualmente; em produção
  a coluna `chuva_mm` da série viria do INMET (pacote diário) ou de um
  pluviômetro local.
