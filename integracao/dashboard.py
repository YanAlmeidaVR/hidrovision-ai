# -*- coding: utf-8 -*-
"""
dashboard.py — HidroVision AI (Fase 4)

Painel de monitoramento. Roda com:

    streamlit run dashboard.py

Duas abas, com propósitos distintos:

  MONITORAMENTO  o estado real: leitura da estação ou da régua, tendência,
                 previsão e alertas gravados. Nada é inventado aqui.

  SIMULAÇÃO      cenários hipotéticos de chuva. Serve para responder "e se
                 chover forte a semana toda?" sem esperar chover, e para
                 demonstrar o alerta antecipado na apresentação.

A separação é deliberada: misturar o observado com o hipotético na mesma
tela é como um painel de alerta perde credibilidade.
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
import clima as C

# ----------------------------------------------------------------------
# paleta (a mesma dos slides)
# ----------------------------------------------------------------------
NAVY = "#0B2239"
TEAL = "#0E7490"
TEAL_CLARO = "#7DD3FC"
AMBER = "#F59E0B"
LARANJA = "#EA580C"
VERMELHO = "#DC2626"
VERDE = "#15803D"
TXT = "#1E293B"
MUT = "#64748B"
LINHA = "#E2E8F0"
FUNDO_CARTAO = "#FBFDFF"

CORES_RISCO = {"normal": VERDE, "atencao": AMBER,
               "alerta": LARANJA, "emergencia": VERMELHO}
NOMES_RISCO = {"normal": "Situação normal", "atencao": "Atenção",
               "alerta": "Alerta", "emergencia": "Emergência"}

# Faixas de intensidade conforme a classificação meteorológica usual.
# A duração de cada cenário é o que o distingue: 50 mm/h por duas horas é
# um temporal; 8 mm/h por um dia inteiro é uma frente estacionada, e é essa
# que costuma encher o rio.
CENARIOS = {
    "Sem chuva": (0.0, 6, "o rio segue apenas a própria recessão"),
    "Garoa persistente": (2.0, 12, "chuva fraca, mas contínua por meio dia"),
    "Chuva moderada": (10.0, 6, "faixa de 5 a 25 mm/h, uma tarde de chuva"),
    "Chuva forte": (30.0, 4, "faixa de 25 a 50 mm/h, quatro horas seguidas"),
    "Temporal": (55.0, 2, "acima de 50 mm/h, curto e intenso"),
    "Frente estacionada": (8.0, 24, "chuva moderada sem trégua por 24 horas"),
    "Semana chuvosa": (12.0, 24, "regime persistente; os modelos enxergam 24 h"),
}

st.set_page_config(page_title="HidroVision AI", page_icon="💧",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .stApp {{ background:#FFFFFF; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}

  section[data-testid="stSidebar"] {{ background:{NAVY}; }}
  section[data-testid="stSidebar"] * {{ color:#CBD5E1; }}
  section[data-testid="stSidebar"] label {{ color:#94A3B8 !important;
      font-size:12px !important; }}

  .hv-marca {{ font-size:23px; font-weight:800; color:#fff;
      letter-spacing:-.4px; margin:2px 0 0; }}
  .hv-sub {{ font-size:11px; color:{TEAL_CLARO}; letter-spacing:1.2px;
      text-transform:uppercase; margin:2px 0 20px; }}

  h1 {{ color:{NAVY} !important; font-weight:800 !important;
      letter-spacing:-1px; font-size:34px !important; }}
  h2 {{ color:{NAVY} !important; font-weight:700 !important;
      letter-spacing:-.4px; font-size:19px !important;
      margin-top:.4rem !important; }}
  h3 {{ color:{NAVY} !important; font-weight:700 !important;
      font-size:16px !important; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {LINHA}; }}
  .stTabs [data-baseweb="tab"] {{ height:44px; padding:0 18px;
      font-weight:600; color:{MUT}; }}
  .stTabs [aria-selected="true"] {{ color:{TEAL} !important; }}

  .hv-card {{ background:{FUNDO_CARTAO}; border:1px solid {LINHA};
      border-radius:12px; padding:18px 20px; height:100%; }}
  .hv-lab {{ font-size:11px; color:{MUT}; text-transform:uppercase;
      letter-spacing:.8px; font-weight:700; }}
  .hv-num {{ font-size:36px; font-weight:800; color:{NAVY};
      line-height:1.15; letter-spacing:-1.4px; margin-top:2px; }}
  .hv-nota {{ font-size:11.5px; color:{MUT}; margin-top:8px;
      line-height:1.45; }}
  .hv-pill {{ display:inline-block; font-size:11px; font-weight:700;
      padding:3px 10px; border-radius:20px; margin-top:8px; }}

  .hv-status {{ border-radius:12px; padding:14px 20px; margin:20px 0 8px;
      border-left:5px solid; font-size:16px; font-weight:700; }}
  .hv-limiar {{ font-size:11.5px; color:{MUT}; letter-spacing:.2px; }}
  .hv-info {{ background:#F8FAFC; border:1px solid {LINHA};
      border-left:4px solid {TEAL}; border-radius:8px; padding:11px 16px;
      font-size:13px; color:{TXT}; margin-bottom:18px; }}
  .hv-legenda {{ font-size:11.5px; color:{MUT}; margin-top:-6px; }}

  hr {{ border-color:{LINHA}; margin:1.6rem 0; }}
</style>
""", unsafe_allow_html=True)


def card(lab, num, nota="", pill=None, pill_cor=TEAL, cor_num=NAVY):
    h = (f'<div class="hv-card"><div class="hv-lab">{lab}</div>'
         f'<div class="hv-num" style="color:{cor_num}">{num}</div>')
    if pill:
        h += (f'<div class="hv-pill" style="background:{pill_cor}1A;'
              f'color:{pill_cor}">{pill}</div>')
    if nota:
        h += f'<div class="hv-nota">{nota}</div>'
    return h + "</div>"


def eixos(y_min, y_max, titulo_y="nível (cm)"):
    ex = alt.X("ts:T", title=None,
               axis=alt.Axis(grid=False, labelColor=MUT, labelFontSize=11,
                             tickColor=LINHA, domainColor=LINHA,
                             labelAngle=0, format="%d/%m %Hh"))
    ey = alt.Y("nivel:Q", title=titulo_y,
               scale=alt.Scale(domain=[y_min, y_max], nice=False),
               axis=alt.Axis(grid=True, gridColor=LINHA, gridDash=[3, 4],
                             labelColor=MUT, labelFontSize=11,
                             titleColor=MUT, titleFontSize=11,
                             tickColor=LINHA, domain=False, tickSize=0))
    return ex, ey


def regras_limiar(limiares, y_min, y_max, so_visiveis=True):
    """Limiares como linhas finas. Fora da escala, não desenha."""
    out = []
    for chave, cor in (("atencao", AMBER), ("alerta", LARANJA),
                       ("emergencia", VERMELHO)):
        v = limiares[chave]
        if so_visiveis and not (y_min <= v <= y_max):
            continue
        base = alt.Chart(pd.DataFrame([{"nivel": v}]))
        esc = alt.Scale(domain=[y_min, y_max], nice=False)
        out.append(base.mark_rule(color=cor, strokeWidth=1,
                                  strokeDash=[4, 4], opacity=0.6)
                   .encode(y=alt.Y("nivel:Q", scale=esc, title=None)))
        out.append(base.mark_text(text=f"{NOMES_RISCO[chave].lower()} {v:.0f}",
                                  align="right", baseline="bottom", dy=-3,
                                  fontSize=10, fontWeight=600, color=cor,
                                  opacity=0.9, x=alt.expr("width - 4"))
                   .encode(y=alt.Y("nivel:Q", scale=esc, title=None)))
    return out


def finalizar(camadas, altura=330):
    return (alt.layer(*camadas).properties(height=altura)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor=TXT, titleColor=MUT, labelFontSize=11,
                              titleFontSize=11, orient="top",
                              direction="horizontal", symbolStrokeWidth=3))


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
# barra lateral: só a fonte de dados
# ----------------------------------------------------------------------
st.sidebar.markdown('<div class="hv-marca">HidroVision AI</div>'
                    '<div class="hv-sub">Painel de monitoramento</div>',
                    unsafe_allow_html=True)

modo = st.sidebar.radio(
    "Ponto monitorado", ["maquete", "estacao"],
    format_func=lambda m: ("Régua urbana · 0 a 100 cm" if m == "maquete"
                           else "Estação 61305000 · cota do rio"))

db = st.sidebar.text_input("Banco de dados",
                           "demo.db" if modo == "maquete" else "hidrovision.db")
pasta_modelos = st.sidebar.text_input("Pasta dos modelos",
                                      "../preditivo/modelos")

st.sidebar.divider()
st.sidebar.subheader("Ciclo automático")
auto = st.sidebar.toggle("Consultar sozinho", value=False,
                         help="O painel repete o ciclo no intervalo escolhido. "
                              "Quando a ANA está fora do ar, ele continua "
                              "tentando até o serviço voltar.")
intervalo_min = st.sidebar.slider("Intervalo (min)", 1, 30, 15,
                                  disabled=not auto)
retry_min = st.sidebar.slider("Nova tentativa se a ANA cair (min)", 1, 15, 3,
                              disabled=not auto,
                              help="O serviço da ANA sai do ar com frequência. "
                                   "Enquanto estiver fora, o painel tenta em "
                                   "intervalo mais curto.")

st.sidebar.divider()
horas_hist = st.sidebar.slider("Janela do gráfico (h)", 6, 168, 48, step=6)
janela_tend = st.sidebar.slider("Janela da tendência (h)", 1, 24,
                                12 if modo == "estacao" else 1)
st.sidebar.caption("A tendência é uma regressão sobre essa janela. Leituras "
                   "horárias pedem janela larga; a câmera, curta.")

if not os.path.exists(db):
    st.title("HidroVision AI")
    st.warning(f"O banco `{db}` ainda não existe. Rode um ciclo de "
               "monitoramento ou a demonstração para criá-lo.")
    st.stop()

p = PL.Pipeline(modo=modo, db=db, pasta_modelos=pasta_modelos,
                prever_a_cada_min=0)
estado = p.estado_atual()

# O estado_atual usa janela fixa de 3 h, dimensionada para a câmera lendo a
# cada 30-60 s. Com a estação publicando de hora em hora, sobra um ponto só.
import tendencia as T
import projecao as PJ
_recente = p.banco.serie_recente(horas=max(janela_tend, 3))
if len(_recente) >= 3:
    estado["tendencia"] = T.calcular(_recente, janela_min=janela_tend * 60)
    _u = estado["ultima_leitura"]
    if _u is not None:
        estado["projecao"] = PJ.projetar(
            _u["nivel_cm"], estado["tendencia"], _recente,
            nivel_critico=estado["nivel_critico"])

critico = estado["nivel_critico"]
limiares = estado["limiares"]
ult = estado["ultima_leitura"]
tend = estado["tendencia"]
proj = estado["projecao"]

st.title("HidroVision AI")

if ult is None:
    st.warning("Nenhuma leitura registrada neste banco ainda.")
    st.stop()

nivel = float(ult["nivel_cm"])
aba_mon, aba_sim = st.tabs(["  Monitoramento  ", "  Simulação de chuva  "])

# ======================================================================
# ABA 1 — MONITORAMENTO
# ======================================================================
with aba_mon:
    esq, dir_ = st.columns([3, 1], gap="large")
    with esq:
        st.markdown("## Estado atual")
    with dir_:
        rodar_ciclo = st.button("Consultar a estação", use_container_width=True,
                                type="primary")

    ana_id = os.environ.get("ANA_ID")
    ana_senha = os.environ.get("ANA_SENHA")

    with st.expander("Opções do ciclo de monitoramento"):
        o1, o2, o3 = st.columns(3)
        dias_ana = o1.slider("Dias a buscar na ANA", 1, 7, 1)
        notificar = o2.checkbox("Enviar ao Telegram", value=False)
        o3.caption("O ciclo consulta a estação e a previsão de chuva, roda os "
                   "modelos e grava o resultado.")
        if not (ana_id and ana_senha):
            st.caption("ANA_ID e ANA_SENHA não estão definidas no ambiente. "
                       "Sem elas o ciclo roda com a série já gravada.")

    def executar_ciclo():
        """Roda um ciclo e registra o resultado no estado da sessão."""
        try:
            mon = M.Monitor(db=db, pasta_modelos=pasta_modelos,
                            ana_id=ana_id, ana_senha=ana_senha,
                            notificar=notificar, dias_historico=dias_ana)
            mon.ciclo(verboso=False)
            fora = mon._ana_fora
            mon.banco.fechar()
            st.session_state["ciclo"] = {
                "quando": pd.Timestamp.now(tz=B.FUSO), "ana_fora": fora}
            # contador de tentativas seguidas com a ANA fora, para o painel
            # dizer há quanto tempo o serviço não responde
            if fora:
                st.session_state["ana_falhas"] = \
                    st.session_state.get("ana_falhas", 0) + 1
                st.session_state.setdefault(
                    "ana_fora_desde", pd.Timestamp.now(tz=B.FUSO))
            else:
                st.session_state["ana_falhas"] = 0
                st.session_state.pop("ana_fora_desde", None)
        except Exception as e:
            st.session_state["ciclo"] = {"erro": str(e),
                                         "quando": pd.Timestamp.now(tz=B.FUSO)}
            st.session_state["ana_falhas"] = \
                st.session_state.get("ana_falhas", 0) + 1
            st.session_state.setdefault(
                "ana_fora_desde", pd.Timestamp.now(tz=B.FUSO))

    if rodar_ciclo:
        with st.spinner("Consultando a estação e rodando os modelos..."):
            executar_ciclo()
        st.rerun()

    ciclo = st.session_state.get("ciclo")
    falhas = st.session_state.get("ana_falhas", 0)

    if ciclo:
        if "erro" in ciclo:
            st.error(f"O ciclo não completou: {ciclo['erro']}")
        else:
            txt = f"Último ciclo às {ciclo['quando']:%d/%m %H:%M}"
            if ciclo.get("ana_fora"):
                desde = st.session_state.get("ana_fora_desde")
                txt += " · serviço da ANA fora do ar"
                if desde is not None:
                    mins = (pd.Timestamp.now(tz=B.FUSO)
                            - desde).total_seconds() / 60
                    txt += (f", sem resposta há {mins:.0f} min "
                            f"({falhas} tentativa{'s' if falhas != 1 else ''})")
                txt += ". As leituras já gravadas continuam válidas."
            st.markdown(f'<div class="hv-info">{txt}</div>',
                        unsafe_allow_html=True)

    # Recarrega sozinho no intervalo escolhido. Quando a ANA está fora, usa o
    # intervalo curto: o serviço volta em minutos, e insistir é o que garante
    # que a série não fique com buraco. É o mesmo princípio da retentativa do
    # monitor, só que visível para quem está olhando a tela.
    if auto:
        espera = retry_min if falhas else intervalo_min
        st.markdown(
            f'<div class="hv-legenda">Ciclo automático ativo · próxima '
            f'consulta em até {espera} min'
            + (f" · em modo de retentativa" if falhas else "")
            + '</div>', unsafe_allow_html=True)
        st.markdown(
            f'<meta http-equiv="refresh" content="{int(espera * 60)}">',
            unsafe_allow_html=True)
        ultimo = (ciclo or {}).get("quando")
        agora_ts = pd.Timestamp.now(tz=B.FUSO)
        if ultimo is None or (agora_ts - ultimo).total_seconds() >= espera * 60:
            with st.spinner("Ciclo automático: consultando a estação..."):
                executar_ciclo()
            st.rerun()

    patamar = "normal"
    for t in ("atencao", "alerta", "emergencia"):
        if nivel >= limiares[t]:
            patamar = t
    cor = CORES_RISCO[patamar]

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.markdown(card(
        "Nível atual", f"{nivel:.0f} cm", cor_num=TEAL,
        nota=f"leitura de {pd.to_datetime(ult['ts']):%d/%m às %H:%M}"),
        unsafe_allow_html=True)

    if tend is not None and tend.taxa_cm_h is not None:
        seta = {"subindo": "↑", "descendo": "↓"}.get(tend.rotulo, "→")
        cor_t = {"subindo": AMBER, "descendo": TEAL}.get(tend.rotulo, MUT)
        c2.markdown(card("Tendência", f"{tend.taxa_cm_h:+.1f} cm/h",
                         pill=f"{seta} {tend.rotulo}", pill_cor=cor_t,
                         nota=f"resolução {getattr(tend,'resolucao','n/d')} · "
                              f"janela de {janela_tend} h"),
                    unsafe_allow_html=True)
    else:
        c2.markdown(card("Tendência", "—",
                         nota=getattr(tend, "detalhe", "sem dados suficientes")
                              if tend else "sem dados"), unsafe_allow_html=True)

    c3.markdown(card(f"Folga até {critico:.0f} cm", f"{critico - nivel:.0f} cm",
                     nota="distância até o ponto em que a água atinge a "
                          "área urbana"), unsafe_allow_html=True)

    if proj is not None and proj.estado == "subindo":
        c4.markdown(card("Atinge o crítico em", proj.tempo_formatado,
                         pill="↑ subindo", pill_cor=AMBER,
                         nota="extrapolação da velocidade atual"),
                    unsafe_allow_html=True)
    elif proj is not None and proj.estado == "descendo":
        c4.markdown(card("Trajetória", "recuando", pill_cor=TEAL,
                         nota="a projeção não se aplica na descida"),
                    unsafe_allow_html=True)
    else:
        c4.markdown(card("Trajetória", "estável",
                         nota="sem subida sustentada, a projeção não se aplica"),
                    unsafe_allow_html=True)

    st.markdown(
        f'<div class="hv-status" style="border-color:{cor};background:{cor}0F;'
        f'color:{cor}">'
        f'{NOMES_RISCO[patamar] if patamar == "normal" else NOMES_RISCO[patamar].upper()}'
        f'{"" if patamar == "normal" else f": nível acima de {limiares[patamar]:.0f} cm"}'
        f'</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hv-limiar">Limiares deste ponto · atenção '
        f'{limiares["atencao"]:.0f} · alerta {limiares["alerta"]:.0f} · '
        f'emergência {limiares["emergencia"]:.0f} · crítico {critico:.0f} cm'
        f'</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # previsão em tabela: o número que interessa, sem depender do gráfico
    # ------------------------------------------------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Previsão do nível")

    ultima_prev = p.banco.ultima_previsao()
    if not ultima_prev or ultima_prev.get("nivel_atual") is None:
        st.caption("Nenhuma previsão registrada. Rode um ciclo de "
                   "monitoramento para calcular.")
    else:
        base_n = float(ultima_prev["nivel_atual"])
        linhas = []
        for h in (6, 12, 24):
            sem = ultima_prev.get(f"prev_{h}h")
            com = ultima_prev.get(f"prev_{h}h_chuva")
            if not isinstance(sem, (int, float)):
                continue
            linha = {
                "horizonte": f"{h} horas",
                "sem chuva nova": f"{sem:.0f} cm",
                "variação": f"{sem - base_n:+.0f} cm",
            }
            if isinstance(com, (int, float)):
                linha["com a chuva prevista"] = f"{com:.0f} cm"
                linha["efeito da chuva"] = f"{com - sem:+.0f} cm"
            else:
                linha["com a chuva prevista"] = "—"
                linha["efeito da chuva"] = "—"
            linhas.append(linha)

        if linhas:
            st.dataframe(pd.DataFrame(linhas), use_container_width=True,
                         hide_index=True)
            quando = ultima_prev.get("ts")
            nota = (f"Calculada às {quando:%d/%m %H:%M} a partir de "
                    f"{base_n:.0f} cm." if quando is not None
                    else f"A partir de {base_n:.0f} cm.")
            st.markdown(
                f'<div class="hv-legenda">{nota} A coluna sem chuva nova '
                f'mostra o rio seguindo a própria recessão; a outra injeta a '
                f'chuva prevista para a bacia. A diferença entre as duas é o '
                f'impacto esperado da precipitação.</div>',
                unsafe_allow_html=True)
        else:
            st.caption("A previsão registrada não trouxe os horizontes.")

    # ------------------------------------------------------------------
    # chuva: bacia alta (alimenta os modelos) e cidade (alerta local)
    # ------------------------------------------------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Chuva")

    ch1, ch2 = st.columns(2, gap="medium")

    medida = p.banco.chuva_recente(horas=6)
    if medida:
        mmh = medida["mmh"]
        acum = medida["acum_mm"]
        nome = C.classificar(mmh)
        grau, _ = C.avaliar_local({"mmh": mmh, "acum_mm": acum,
                                   "horas_acum": 6, "intensidade": nome})
        cor_ch = CORES_RISCO.get(grau, MUT)
        ch1.markdown(card(
            "Chuva na cidade · medida",
            f"{mmh:.1f} mm/h",
            pill=nome if mmh >= 0.2 else "sem chuva",
            pill_cor=cor_ch if grau != "normal" else MUT,
            nota=f"{acum:.1f} mm nas últimas 6 h · pluviômetro da estação · "
                 f"leitura de {medida['ts']:%d/%m %H:%M}"),
            unsafe_allow_html=True)
    else:
        ch1.markdown(card(
            "Chuva na cidade · medida", "—",
            nota="sem dado de pluviômetro no banco. Rode um ciclo para trazer "
                 "a chuva medida pela estação."), unsafe_allow_html=True)

    prev_bacia = p.banco.ultima_previsao()
    if prev_bacia and prev_bacia.get("chuva_total_mm") is not None:
        tot = float(prev_bacia["chuva_total_mm"])
        pico = prev_bacia.get("chuva_pico_mmh")
        prob = prev_bacia.get("chuva_prob_max")
        ch2.markdown(card(
            "Bacia alta · previsão 24 h", f"{tot:.0f} mm",
            pill=(f"pico {pico:.0f} mm/h" if isinstance(pico, (int, float))
                  else None), pill_cor=TEAL,
            nota=f"Maria da Fé, a montante · probabilidade máxima de "
                 f"{prob or 0}%. É esta a chuva que alimenta os modelos: "
                 f"ela ainda vai escoar até aqui."),
            unsafe_allow_html=True)
    else:
        ch2.markdown(card(
            "Bacia alta · previsão 24 h", "—",
            nota="sem previsão registrada. Rode um ciclo de monitoramento."),
            unsafe_allow_html=True)

    st.markdown(
        '<div class="hv-legenda">A chuva da bacia entra nos modelos do rio, '
        'com horas de atraso até chegar aqui. A da cidade não passa por eles: '
        'vai para a drenagem urbana e alaga rua antes de qualquer cheia.'
        '</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Histórico")

    serie = p.banco.serie_recente(horas=horas_hist)
    if serie.empty:
        st.caption("Sem leituras na janela selecionada.")
    else:
        hist = (serie[["ts", "nivel_cm"]].rename(columns={"nivel_cm": "nivel"})
                .assign(ts=lambda d: pd.to_datetime(d["ts"])))

        prev = estado.get("previsoes")
        futuro = pd.DataFrame()
        if prev:
            base_ts = hist["ts"].max()
            pts = [{"ts": base_ts, "nivel": float(hist["nivel"].iloc[-1])}]
            for h in (6, 12, 24):
                v = prev.get(f"{h}h")
                if isinstance(v, (int, float)):
                    pts.append({"ts": base_ts + pd.Timedelta(hours=h),
                                "nivel": float(v)})
            if len(pts) > 1:
                futuro = pd.DataFrame(pts)

        todos = pd.concat([hist, futuro]) if not futuro.empty else hist
        margem = max((todos["nivel"].max() - todos["nivel"].min()) * 0.35, 4)
        y_min = todos["nivel"].min() - margem
        y_max = todos["nivel"].max() + margem
        ex, ey = eixos(y_min, y_max)

        camadas = [
            alt.Chart(hist).mark_line(color=TEAL, strokeWidth=2.4,
                                      interpolate="monotone")
            .encode(x=ex, y=ey,
                    tooltip=[alt.Tooltip("ts:T", title="quando",
                                         format="%d/%m %H:%M"),
                             alt.Tooltip("nivel:Q", title="nível",
                                         format=".1f")])
        ]
        if not futuro.empty:
            camadas.append(
                alt.Chart(futuro).mark_line(color=TEAL, strokeWidth=1.8,
                                            strokeDash=[5, 4], opacity=0.65)
                .encode(x=ex, y=ey))
            camadas.append(
                alt.Chart(futuro.iloc[1:]).mark_point(
                    color=TEAL, size=42, filled=True, opacity=0.8)
                .encode(x=ex, y=ey,
                        tooltip=[alt.Tooltip("ts:T", title="horizonte",
                                             format="%d/%m %H:%M"),
                                 alt.Tooltip("nivel:Q", title="previsto",
                                             format=".1f")]))
        camadas += regras_limiar(limiares, y_min, y_max)

        st.altair_chart(finalizar(camadas), use_container_width=True)
        st.markdown(
            '<div class="hv-legenda">Linha cheia: nível medido. '
            + ("Tracejada: previsão para 6, 12 e 24 horas. "
               if not futuro.empty else "")
            + "Só os limiares dentro da escala são desenhados.</div>",
            unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Alertas registrados")

    al = p.banco.alertas_recentes(30)
    if al.empty:
        st.caption("Nenhum alerta registrado neste banco.")
    else:
        al = al.copy()
        tabela = pd.DataFrame({
            "quando": pd.to_datetime(al["ts"]).dt.strftime("%d/%m %H:%M"),
            "categoria": al["origem"],
            "grau": al["tipo"].map(lambda t: NOMES_RISCO.get(t, t)),
            "estado": al["estado"],
            "nível": al["nivel_cm"].map(lambda v: f"{v:.0f} cm"),
            "mensagem": al["mensagem"],
        })
        st.dataframe(tabela, use_container_width=True, hide_index=True,
                     height=min(360, 40 + 35 * len(tabela)))

# ======================================================================
# ABA 2 — SIMULAÇÃO
# ======================================================================
with aba_sim:
    st.markdown("## Cenários de chuva")
    st.markdown(
        '<div class="hv-info">Nada aqui é medição. São cenários hipotéticos '
        'rodados nos modelos a partir do nível atual, para responder o que '
        'aconteceria se chovesse de um jeito ou de outro.</div>',
        unsafe_allow_html=True)

    if p.preditor is None:
        st.warning("Modelos não carregados. Verifique a pasta dos modelos na "
                   "barra lateral.")
    elif modo == "maquete":
        st.warning("Os modelos foram treinados na cota do rio, que varia de 14 "
                   "a 447 cm. A régua urbana opera de 0 a 100 cm, outra escala "
                   "e outra física. Troque para a estação para simular.")
    else:
        e1, e2 = st.columns([2, 3], gap="large")

        with e1:
            nome = st.radio("Cenário", list(CENARIOS.keys()), index=3)
            mmh_pad, horas_pad, desc = CENARIOS[nome]
            st.caption(desc)

            ajustar = st.checkbox("Ajustar manualmente")
            if ajustar:
                mmh = st.slider("Intensidade (mm/h)", 0.0, 80.0,
                                float(mmh_pad), step=1.0)
                horas = st.slider("Duração (h)", 1, 24, int(horas_pad))
            else:
                mmh, horas = mmh_pad, horas_pad
                st.markdown(
                    f'<div class="hv-legenda">{mmh:.0f} mm/h durante {horas} h '
                    f'· total de {mmh * horas:.0f} mm</div>',
                    unsafe_allow_html=True)

            rodar_sim = st.button("Rodar cenário", use_container_width=True,
                                  type="primary")

        with e2:
            if not rodar_sim:
                st.caption("Escolha um cenário à esquerda e toque em Rodar "
                           "cenário. O resultado compara a previsão com e sem "
                           "a chuva escolhida.")
            else:
                with st.spinner("Rodando os modelos..."):
                    base = p.preditor.prever(
                        p.banco.serie_horaria(horas=30 * 24))
                    sim = (p.simular_chuva(mmh, horas_de_chuva=horas)
                           if mmh > 0 else base)

                if not sim or not base:
                    st.warning("A série histórica não tem 25 horas contíguas, "
                               "que é o mínimo para as defasagens do modelo. "
                               "Rode um ciclo de monitoramento para trazer "
                               "mais horas da estação.")
                else:
                    atual = float(sim.get("nivel_atual", nivel))
                    cs = st.columns(3, gap="medium")
                    for c, h in zip(cs, (6, 12, 24)):
                        v = sim.get(f"{h}h")
                        vb = base.get(f"{h}h")
                        if not isinstance(v, (int, float)):
                            continue
                        efeito = (v - vb) if isinstance(vb, (int, float)) else 0
                        cor_p = (AMBER if efeito > 0.5
                                 else TEAL if efeito < -0.5 else MUT)
                        c.markdown(card(
                            f"em {h} horas", f"{v:.0f} cm",
                            pill=f"{efeito:+.0f} cm pela chuva", pill_cor=cor_p,
                            nota=f"variação de {v - atual:+.0f} cm sobre o "
                                 f"nível atual"), unsafe_allow_html=True)

                    # curva comparativa
                    agora = pd.Timestamp.now(tz=B.FUSO).floor("h")
                    linhas = []
                    for rot, d in (("sem a chuva", base), ("com a chuva", sim)):
                        linhas.append({"ts": agora, "nivel": atual,
                                       "cenario": rot})
                        for h in (6, 12, 24):
                            v = d.get(f"{h}h")
                            if isinstance(v, (int, float)):
                                linhas.append(
                                    {"ts": agora + pd.Timedelta(hours=h),
                                     "nivel": float(v), "cenario": rot})
                    cmp_df = pd.DataFrame(linhas)

                    margem = max((cmp_df["nivel"].max()
                                  - cmp_df["nivel"].min()) * 0.4, 4)
                    y_min = cmp_df["nivel"].min() - margem
                    y_max = cmp_df["nivel"].max() + margem
                    ex, ey = eixos(y_min, y_max)

                    cor_cen = alt.Color(
                        "cenario:N", title=None,
                        scale=alt.Scale(domain=["sem a chuva", "com a chuva"],
                                        range=[MUT, TEAL]))
                    camadas = [
                        alt.Chart(cmp_df).mark_line(strokeWidth=2.4,
                                                    interpolate="monotone")
                        .encode(x=ex, y=ey, color=cor_cen,
                                strokeDash=alt.StrokeDash(
                                    "cenario:N", legend=None,
                                    scale=alt.Scale(
                                        domain=["sem a chuva", "com a chuva"],
                                        range=[[5, 4], [1, 0]]))),
                        alt.Chart(cmp_df).mark_point(size=45, filled=True)
                        .encode(x=ex, y=ey, color=cor_cen,
                                tooltip=[alt.Tooltip("cenario:N", title="cenário"),
                                         alt.Tooltip("ts:T", title="quando",
                                                     format="%d/%m %Hh"),
                                         alt.Tooltip("nivel:Q", title="nível",
                                                     format=".1f")]),
                    ]
                    camadas += regras_limiar(limiares, y_min, y_max)
                    st.altair_chart(finalizar(camadas, altura=290),
                                    use_container_width=True)

                    total = mmh * horas
                    st.markdown(
                        f'<div class="hv-legenda">Cenário: {mmh:.0f} mm/h por '
                        f'{horas} h, total de {total:.0f} mm. Os modelos '
                        f'preveem a variação do nível; o valor exibido é o '
                        f'nível atual somado a ela.</div>',
                        unsafe_allow_html=True)

                    if total >= 100:
                        st.markdown(
                            '<div class="hv-info">Acumulados muito acima do '
                            'que ocorreu no período de treino levam o modelo a '
                            'extrapolar. O número continua sendo uma estimativa, '
                            'mas com incerteza maior do que a medida em '
                            'validação.</div>', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### O que o horizonte cobre")
        st.caption(
            "Os três modelos foram treinados para 6, 12 e 24 horas à frente. "
            "Um cenário de semana inteira é representado pelo regime de chuva "
            "mantido nas 24 horas seguintes: além disso, a previsão sairia do "
            "domínio em que os modelos foram validados.")

try:
    p.banco.fechar()
except Exception:
    pass