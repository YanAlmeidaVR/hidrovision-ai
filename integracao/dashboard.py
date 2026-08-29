# -*- coding: utf-8 -*-
"""
dashboard.py — HidroVision AI (Fase 4)

Painel de monitoramento. Roda com:

    streamlit run dashboard.py

O visual segue a identidade do projeto: fundo claro, cartões em azul-gelo,
painéis escuros em azul-marinho e o teal como cor de ação. Os gráficos são
desenhados em Altair para permitir o mesmo tratamento dos slides — limiares
como linhas finas tracejadas, e não como séries competindo com a leitura.

O painel lê o banco e, pelo botão da barra lateral, também executa o ciclo
de monitoramento (consulta a ANA, roda os modelos, grava e alerta).
"""
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import banco as B
import pipeline as PL
import monitor as M

# ----------------------------------------------------------------------
# identidade visual (mesma paleta dos slides)
# ----------------------------------------------------------------------
NAVY = "#0B2239"
NAVY2 = "#123A5C"
TEAL = "#0E7490"
TEAL_D = "#155E75"
ICE = "#E0F2FE"
ICE2 = "#F0F9FF"
AMBER = "#F59E0B"
LARANJA = "#EA580C"
VERMELHO = "#DC2626"
VERDE = "#15803D"
TXT = "#1E293B"
MUT = "#64748B"
GRID = "#E2E8F0"

CORES_RISCO = {"normal": VERDE, "atencao": AMBER,
               "alerta": LARANJA, "emergencia": VERMELHO}
ROTULOS = {"atencao": "Atenção", "alerta": "Alerta",
           "emergencia": "Emergência", "normal": "Normal"}

st.set_page_config(page_title="HidroVision AI", page_icon="💧",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background: #FFFFFF; }}
  section[data-testid="stSidebar"] {{
      background: {NAVY};
  }}
  section[data-testid="stSidebar"] * {{ color: #DCEFFB; }}
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {{ color: #FFFFFF; }}

  .hv-marca {{ font-size: 25px; font-weight: 800; color: #FFFFFF;
               letter-spacing: -.4px; margin: 4px 0 0; }}
  .hv-marca-sub {{ font-size: 12.5px; color: #7DD3FC; margin: 0 0 18px;
                   letter-spacing: .3px; text-transform: uppercase; }}

  h1 {{ color: {NAVY} !important; font-weight: 800 !important;
        letter-spacing: -.8px; }}
  h2, h3 {{ color: {NAVY} !important; font-weight: 700 !important;
            letter-spacing: -.3px; }}

  .hv-cartao {{
      background: {ICE2}; border: 1px solid {GRID}; border-radius: 10px;
      padding: 16px 18px; height: 100%;
  }}
  .hv-rotulo {{ font-size: 12px; color: {MUT}; text-transform: uppercase;
                letter-spacing: .6px; font-weight: 600; margin-bottom: 2px; }}
  .hv-valor {{ font-size: 34px; font-weight: 800; color: {NAVY};
               line-height: 1.1; letter-spacing: -1px; }}
  .hv-valor-teal {{ color: {TEAL}; }}
  .hv-nota {{ font-size: 12px; color: {MUT}; margin-top: 6px;
              line-height: 1.4; }}
  .hv-tag {{ display: inline-block; font-size: 11.5px; font-weight: 700;
             padding: 2px 9px; border-radius: 20px; margin-top: 6px; }}

  .hv-faixa {{ border-radius: 10px; padding: 15px 20px; color: #fff;
               font-size: 18px; font-weight: 700; margin: 18px 0 6px; }}
  .hv-limiares {{ font-size: 12.5px; color: {MUT}; margin-bottom: 6px; }}
  .hv-ciclo {{ background: {NAVY}; color: #DCEFFB; border-radius: 10px;
               padding: 12px 18px; font-size: 13.5px; margin-bottom: 16px; }}
  .hv-ciclo b {{ color: #7DD3FC; }}

  hr {{ border-color: {GRID}; }}
  [data-testid="stMetricValue"] {{ color: {NAVY}; }}
</style>
""", unsafe_allow_html=True)


def cartao(rotulo, valor, nota="", tag=None, tag_cor=TEAL, teal=False):
    """Cartão de métrica no mesmo desenho dos slides."""
    cls = "hv-valor hv-valor-teal" if teal else "hv-valor"
    html = (f'<div class="hv-cartao">'
            f'<div class="hv-rotulo">{rotulo}</div>'
            f'<div class="{cls}">{valor}</div>')
    if tag:
        html += (f'<div class="hv-tag" style="background:{tag_cor}22;'
                 f'color:{tag_cor}">{tag}</div>')
    if nota:
        html += f'<div class="hv-nota">{nota}</div>'
    return html + "</div>"


def descrever_clima(clima):
    if not clima:
        return "previsão de chuva indisponível"
    try:
        import clima as C
        return C.descrever(clima)
    except Exception:
        mm = clima.get("total_mm")
        return (f"{mm:.0f} mm previstos em 24 h"
                if isinstance(mm, (int, float)) else "previsão obtida")


# ----------------------------------------------------------------------
# barra lateral
# ----------------------------------------------------------------------
st.sidebar.markdown('<div class="hv-marca">HidroVision AI</div>'
                    '<div class="hv-marca-sub">Painel de monitoramento</div>',
                    unsafe_allow_html=True)

modo = st.sidebar.radio(
    "Ponto monitorado",
    ["maquete", "estacao"],
    format_func=lambda m: ("Régua urbana (0 a 100 cm)" if m == "maquete"
                           else "Estação 61305000 (cota do rio)"),
    help=("A régua urbana fica junto à cidade: 100 cm é o ponto em que a água "
          "chega. A estação mede a cota do rio, em outra escala."),
)

db = st.sidebar.text_input("Banco de dados",
                           "demo.db" if modo == "maquete" else "hidrovision.db")
pasta_modelos = st.sidebar.text_input("Pasta dos modelos", "../preditivo/modelos")
horas_hist = st.sidebar.slider("Janela do gráfico (h)", 6, 168, 48, step=6)

st.sidebar.divider()
st.sidebar.subheader("Ciclo de monitoramento")

ana_id = os.environ.get("ANA_ID")
ana_senha = os.environ.get("ANA_SENHA")
if not (ana_id and ana_senha):
    st.sidebar.info("Defina ANA_ID e ANA_SENHA no ambiente para consultar a "
                    "estação. Sem elas, o ciclo roda com a previsão de chuva "
                    "e a série já gravada.")

dias_ana = st.sidebar.slider("Dias a buscar na ANA", 1, 7, 3)
notificar = st.sidebar.checkbox("Enviar alerta ao Telegram", value=False)

if st.sidebar.button("Consultar agora", use_container_width=True,
                     type="primary"):
    with st.spinner("Consultando a estação e rodando os modelos..."):
        try:
            mon = M.Monitor(db=db, pasta_modelos=pasta_modelos,
                            ana_id=ana_id, ana_senha=ana_senha,
                            notificar=notificar, dias_historico=dias_ana)
            risco, prev, clima = mon.ciclo(verboso=False)
            mon.banco.fechar()
            st.session_state["ciclo"] = {
                "quando": pd.Timestamp.now(tz=B.FUSO), "clima": clima,
                "ana_fora": mon._ana_fora,
            }
        except Exception as e:
            st.session_state["ciclo"] = {"erro": str(e)}
    st.rerun()

if st.sidebar.button("Recarregar a tela", use_container_width=True):
    st.rerun()

if not os.path.exists(db):
    st.title("Monitoramento de nível")
    st.warning(f"O banco `{db}` ainda não existe. Rode o ciclo de "
               "monitoramento ou a demonstração para criá-lo.")
    st.stop()

p = PL.Pipeline(modo=modo, db=db, pasta_modelos=pasta_modelos,
                prever_a_cada_min=0)
estado = p.estado_atual()

# O estado_atual calcula a tendência sobre 3 h, janela dimensionada para a
# câmera, que lê a cada 30-60 s. A estação publica de hora em hora: nessa
# janela cabe um ponto só. No modo estação, recalcula com janela maior.
if modo == "estacao":
    import tendencia as T
    import projecao as PJ
    st.sidebar.divider()
    janela_h = st.sidebar.slider("Janela da tendência (h)", 3, 24, 12)
    recente = p.banco.serie_recente(horas=janela_h)
    if len(recente) >= 3:
        estado["tendencia"] = T.calcular(recente, janela_min=janela_h * 60)
        u = estado["ultima_leitura"]
        if u is not None:
            estado["projecao"] = PJ.projetar(
                u["nivel_cm"], estado["tendencia"], recente,
                nivel_critico=estado["nivel_critico"])

critico = estado["nivel_critico"]
limiares = estado["limiares"]
ult = estado["ultima_leitura"]
tend = estado["tendencia"]
proj = estado["projecao"]

# ----------------------------------------------------------------------
# cabeçalho e status
# ----------------------------------------------------------------------
st.title("Monitoramento de nível")

ciclo = st.session_state.get("ciclo")
if ciclo:
    if "erro" in ciclo:
        st.error(f"O ciclo não completou: {ciclo['erro']}")
    else:
        partes = [f"<b>Ciclo executado às {ciclo['quando']:%d/%m %H:%M}</b>"]
        if ciclo.get("ana_fora"):
            partes.append("serviço da ANA fora do ar, sem leitura nova")
        partes.append(descrever_clima(ciclo.get("clima")))
        st.markdown(f'<div class="hv-ciclo">{" · ".join(partes)}</div>',
                    unsafe_allow_html=True)

if ult is None:
    st.warning("Nenhuma leitura registrada neste banco ainda.")
    st.stop()

nivel = float(ult["nivel_cm"])
folga = critico - nivel

patamar = "normal"
for t in ("atencao", "alerta", "emergencia"):
    if nivel >= limiares[t]:
        patamar = t

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    st.markdown(cartao(
        "Nível atual", f"{nivel:.0f} cm",
        nota=f"leitura de {pd.to_datetime(ult['ts']):%d/%m %H:%M} · "
             f"método {ult.get('metodo') or 'n/d'}",
        teal=True), unsafe_allow_html=True)

with c2:
    if tend is not None and tend.taxa_cm_h is not None:
        seta = {"subindo": "↑", "descendo": "↓"}.get(tend.rotulo, "→")
        cor_t = {"subindo": AMBER, "descendo": TEAL}.get(tend.rotulo, MUT)
        st.markdown(cartao(
            "Tendência", f"{tend.taxa_cm_h:+.1f} cm/h",
            tag=f"{seta} {tend.rotulo}", tag_cor=cor_t,
            nota=f"resolução {getattr(tend, 'resolucao', 'n/d')}"),
            unsafe_allow_html=True)
    else:
        st.markdown(cartao(
            "Tendência", "indefinida",
            nota=getattr(tend, "detalhe", "sem dados suficientes")
                 if tend else "sem dados"), unsafe_allow_html=True)

with c3:
    st.markdown(cartao(
        f"Folga até {critico:.0f} cm", f"{folga:.0f} cm",
        nota="quanto falta para a água atingir o ponto crítico"),
        unsafe_allow_html=True)

with c4:
    if proj is not None and proj.estado == "subindo":
        st.markdown(cartao(
            "Atinge o crítico em", proj.tempo_formatado,
            tag="↑ subindo", tag_cor=AMBER, nota=proj.detalhe or ""),
            unsafe_allow_html=True)
    elif proj is not None and proj.estado == "descendo":
        st.markdown(cartao("Trajetória", "recuando", tag_cor=TEAL,
                           nota="a projeção não se aplica na descida"),
                    unsafe_allow_html=True)
    else:
        st.markdown(cartao("Trajetória", "estável",
                           nota="sem subida sustentada, projeção não se aplica"),
                    unsafe_allow_html=True)

cor = CORES_RISCO[patamar]
texto = ("Situação normal" if patamar == "normal"
         else f"{ROTULOS[patamar].upper()}: nível acima do limiar de "
              f"{limiares[patamar]:.0f} cm")
st.markdown(f'<div class="hv-faixa" style="background:{cor}">{texto}</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="hv-limiares">Atenção {limiares["atencao"]:.0f} cm · '
    f'Alerta {limiares["alerta"]:.0f} cm · '
    f'Emergência {limiares["emergencia"]:.0f} cm · '
    f'Crítico {critico:.0f} cm</div>', unsafe_allow_html=True)

st.divider()

# ----------------------------------------------------------------------
# gráfico
# ----------------------------------------------------------------------
st.subheader("Histórico e previsão")

serie = p.banco.serie_recente(horas=horas_hist)
if serie.empty:
    st.caption("Sem leituras na janela selecionada.")
else:
    serie = serie.copy()
    serie["ts"] = pd.to_datetime(serie["ts"])
    hist = serie[["ts", "nivel_cm"]].rename(columns={"nivel_cm": "nivel"})
    hist["serie"] = "Nível lido"

    prev = estado.get("previsoes")
    futuro = pd.DataFrame()
    if prev:
        base_ts = serie["ts"].max()
        pontos = [{"ts": base_ts, "nivel": float(serie["nivel_cm"].iloc[-1])}]
        for h in (6, 12, 24):
            v = prev.get(f"{h}h")
            if isinstance(v, (int, float)):
                pontos.append({"ts": base_ts + pd.Timedelta(hours=h),
                               "nivel": float(v)})
        if len(pontos) > 1:
            futuro = pd.DataFrame(pontos)
            futuro["serie"] = "Previsão"

    dados = pd.concat([hist, futuro], ignore_index=True) if not futuro.empty else hist
    y_min = min(dados["nivel"].min(), limiares["atencao"]) * 0.92
    y_max = max(dados["nivel"].max(), limiares["emergencia"]) * 1.06

    eixo_y = alt.Y("nivel:Q", title="nível (cm)",
                   scale=alt.Scale(domain=[y_min, y_max], nice=False),
                   axis=alt.Axis(grid=True, gridColor=GRID, gridDash=[2, 3],
                                 labelColor=MUT, titleColor=MUT,
                                 tickColor=GRID, domain=False))
    eixo_x = alt.X("ts:T", title=None,
                   axis=alt.Axis(grid=False, labelColor=MUT, tickColor=GRID,
                                 domainColor=GRID, format="%d/%m %H:%M"))

    area = (alt.Chart(hist).mark_area(
                line=False, opacity=0.14,
                color=alt.Gradient(
                    gradient="linear", x1=1, x2=1, y1=0, y2=1,
                    stops=[alt.GradientStop(color=TEAL, offset=0),
                           alt.GradientStop(color="#FFFFFF", offset=1)]))
            .encode(x=eixo_x, y=alt.Y("nivel:Q",
                                      scale=alt.Scale(domain=[y_min, y_max],
                                                      nice=False), title=None)))

    linha = (alt.Chart(hist).mark_line(color=TEAL, strokeWidth=2.6,
                                       interpolate="monotone")
             .encode(x=eixo_x, y=eixo_y,
                     tooltip=[alt.Tooltip("ts:T", title="quando",
                                          format="%d/%m %H:%M"),
                              alt.Tooltip("nivel:Q", title="nível (cm)",
                                          format=".1f")]))

    camadas = [area, linha]

    if not futuro.empty:
        camadas.append(
            alt.Chart(futuro).mark_line(color=NAVY2, strokeWidth=2,
                                        strokeDash=[6, 4])
            .encode(x=eixo_x, y=eixo_y,
                    tooltip=[alt.Tooltip("ts:T", title="horizonte",
                                         format="%d/%m %H:%M"),
                             alt.Tooltip("nivel:Q", title="previsto (cm)",
                                         format=".1f")]))
        camadas.append(
            alt.Chart(futuro).mark_point(color=NAVY2, size=45, filled=True)
            .encode(x=eixo_x, y=eixo_y))

    # limiares: linhas finas tracejadas, com o rótulo à esquerda. Não são
    # séries de dados e não devem competir visualmente com a leitura.
    lim_df = pd.DataFrame([
        {"y": limiares["atencao"], "rot": "atenção", "cor": AMBER},
        {"y": limiares["alerta"], "rot": "alerta", "cor": LARANJA},
        {"y": limiares["emergencia"], "rot": "emergência", "cor": VERMELHO},
    ])
    for _, r in lim_df.iterrows():
        base = alt.Chart(pd.DataFrame([{"y": r["y"]}]))
        camadas.append(base.mark_rule(color=r["cor"], strokeWidth=1.2,
                                      strokeDash=[5, 4], opacity=0.75)
                       .encode(y=alt.Y("y:Q", scale=alt.Scale(
                           domain=[y_min, y_max], nice=False), title=None)))
        camadas.append(base.mark_text(
            text=f"{r['rot']} {r['y']:.0f}", align="left", baseline="bottom",
            dx=4, dy=-3, fontSize=10.5, fontWeight="bold", color=r["cor"])
            .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[y_min, y_max],
                                                   nice=False), title=None)))

    st.altair_chart(
        alt.layer(*camadas).properties(height=360)
           .configure_view(strokeWidth=0)
           .configure_axis(labelFontSize=11, titleFontSize=11),
        use_container_width=True)

    legenda = ("Linha cheia: nível lido. " +
               ("Linha tracejada: previsão para 6, 12 e 24 h. "
                if not futuro.empty else "") +
               "As linhas horizontais marcam os limiares de alerta.")
    st.caption(legenda)

st.divider()

# ----------------------------------------------------------------------
# simulação de chuva
# ----------------------------------------------------------------------
st.subheader("Simulação de chuva")

if p.preditor is None:
    st.caption("Modelos não carregados. Confira a pasta dos modelos na barra "
               "lateral.")
elif modo == "maquete":
    st.caption("Os modelos foram treinados na cota do rio, entre 14 e 447 cm. "
               "Na régua urbana, de 0 a 100 cm, a previsão não se aplica: "
               "troque para a estação para simular.")
else:
    col_a, col_b = st.columns([1, 2], gap="large")
    with col_a:
        mmh = st.slider("Intensidade (mm/h)", 0.0, 25.0, 8.0, step=0.5)
        horas = st.slider("Duração (h)", 1, 24, 6)
        rodar = st.button("Simular", use_container_width=True)
    with col_b:
        if rodar:
            with st.spinner("Rodando os modelos..."):
                sim = p.simular_chuva(mmh, horas_de_chuva=horas)
            if not sim:
                st.warning("A série histórica é curta demais para as "
                           "defasagens do modelo. Rode o ciclo para trazer "
                           "mais horas da estação.")
            else:
                atual = sim.get("nivel_atual", nivel)
                cols = st.columns(3, gap="medium")
                for c, chave, rot in zip(cols, ("6h", "12h", "24h"),
                                         ("em 6 horas", "em 12 horas",
                                          "em 24 horas")):
                    v = sim.get(chave)
                    if isinstance(v, (int, float)):
                        d = v - atual
                        cor_d = AMBER if d > 0.5 else (TEAL if d < -0.5 else MUT)
                        c.markdown(cartao(
                            rot, f"{v:.0f} cm",
                            tag=f"{d:+.0f} cm", tag_cor=cor_d),
                            unsafe_allow_html=True)
                st.caption(f"Cenário com {mmh:.1f} mm/h durante {horas} h. "
                           "Os modelos preveem a variação; o valor mostrado é "
                           "o nível atual somado a ela.")
        else:
            st.caption("Ajuste a chuva e toque em Simular para ver o efeito "
                       "sobre a previsão de nível.")

st.divider()

# ----------------------------------------------------------------------
# alertas
# ----------------------------------------------------------------------
st.subheader("Alertas registrados")

al = p.banco.alertas_recentes(30)
if al.empty:
    st.caption("Nenhum alerta registrado neste banco.")
else:
    al = al.copy()
    al["ts"] = pd.to_datetime(al["ts"]).dt.strftime("%d/%m %H:%M")
    tabela = pd.DataFrame({
        "quando": al["ts"],
        "categoria": al["origem"],
        "grau": al["tipo"].map(lambda t: ROTULOS.get(t, t)),
        "estado": al["estado"],
        "nível": al["nivel_cm"].map(lambda v: f"{v:.0f} cm"),
        "mensagem": al["mensagem"],
    })
    st.dataframe(tabela, use_container_width=True, hide_index=True,
                 height=min(400, 42 + 35 * len(tabela)))

st.caption("O painel lê o banco continuamente. Use Consultar agora para "
           "executar um ciclo de monitoramento.")

try:
    p.banco.fechar()
except Exception:
    pass