# -*- coding: utf-8 -*-
"""
Régua da maquete ao vivo, mostrada pelo dashboard quando o ponto monitorado é
a régua urbana.

Não há cheia real numa feira, então a previsão roda sobre um cenário de rio
simulado (modelos XGBoost reais) e a variação prevista é somada à leitura da
régua pra dizer se e quando a água transbordaria. O monitor do rio real fica
no outro ponto monitorado, a estação.
"""
import sys
import time
from pathlib import Path

import cv2
import streamlit as st

INTEGRACAO = Path(__file__).resolve().parent
RAIZ = INTEGRACAO.parent
sys.path.insert(0, str(INTEGRACAO))
sys.path.insert(0, str(RAIZ / "visao"))

import mdYOLO as V
import demo_maquete as D
import camera_rede as CR

WINDOWS = sys.platform.startswith("win")
NCNN = RAIZ / "visao" / "hidrovision_v07_regua_ncnn_model"
PT = RAIZ / "visao" / "hidrovision_v07_regua.pt"
MODELOS_RIO = RAIZ / "preditivo" / "modelos"
CORES = {None: "#15803D", "atencao": "#F59E0B", "alerta": "#EA580C",
         "emergencia": "#DC2626"}
NOMES = {None: "Situação normal", "atencao": "Atenção", "alerta": "Alerta",
         "emergencia": "Emergência"}

REDE = "Raspberry pela rede"
LOCAL = "Câmera deste computador"


@st.cache_resource
def carregar_modelo(caminho):
    from ultralytics import YOLO
    modelo = YOLO(caminho)
    V.montar_paleta(modelo.names)
    return modelo


@st.cache_resource
def camera(indice):
    cap = V.abrir_camera(indice)
    if cap is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def sessao(telegram):
    ss = st.session_state
    if "rio" not in ss:
        ss.rio = D.RioSimulado(str(MODELOS_RIO))
        ss.rio.prever()
        ss.filtro = V.FiltroMediana()
        ss.mensagens = []
    if ss.get("telegram_ativo") != telegram or "alerta" not in ss:
        ss.alerta = D.AlertaPrevisao(D.canais_demo(telegram))
        ss.telegram_ativo = telegram
    return ss


def mostrar_imagem(img):
    try:
        st.image(img, width="stretch")
    except (TypeError, Exception):
        st.image(img, use_container_width=True)


def botao(col, rotulo, **kw):
    try:
        return col.button(rotulo, width="stretch", **kw)
    except TypeError:
        return col.button(rotulo, use_container_width=True, **kw)


def selo(urgencia):
    cor = CORES[urgencia]
    return (f'<div style="background:{cor};color:#fff;border-radius:10px;'
            f'padding:10px 14px;font-weight:700;font-size:20px;text-align:center">'
            f'{NOMES[urgencia]}</div>')


def modelo_padrao():
    """Prefere o V07; o V06 fica como reserva caso o V07 não esteja presente."""
    for versao in ("v07", "v06"):
        tflite = sorted((RAIZ / "visao").rglob(f"*{versao}*.tflite"))
        if tflite:
            return str(tflite[0])
    if NCNN.exists():
        return str(NCNN)
    return str(PT)


def consultar(endereco):
    """Leitura da Raspberry, ou None se ela não responder a tempo."""
    try:
        return CR.buscar_leitura(endereco, timeout=1.5)
    except Exception:
        return None


def mostrar_info(ss, nivel, vals, detalhe):
    if nivel is None:
        st.markdown("### Procurando a régua")
        st.caption(detalhe)
        return

    folga = D.NIVEL_CRITICO - nivel
    m1, m2 = st.columns(2)
    m1.metric("Nível na régua", f"{nivel:.0f} cm")
    m2.metric("Folga até transbordar", f"{folga:.0f} cm")
    if vals:
        st.caption("números detectados: " + " ".join(map(str, vals)))

    aval = D.avaliar_previsao_local(nivel, ss.rio.variacoes())
    urgencia = aval["urgencia"] if aval else None
    st.markdown(selo(urgencia), unsafe_allow_html=True)

    st.markdown("#### Previsão do rio")
    if ss.rio.chuva_prevista_mm:
        st.caption(f"chuva na bacia: {ss.rio.chuva_prevista_mm} mm em 12 h "
                   "(cenário simulado)")
    else:
        st.caption("tempo firme, sem chuva prevista (cenário simulado)")

    if aval:
        linhas = []
        for h, d in sorted(ss.rio.variacoes().items()):
            v = aval["projecao"][h]
            marca = " · **transborda**" if v >= D.NIVEL_CRITICO else ""
            linhas.append(f"| {h} h | {d:+.0f} cm | {v:.0f} cm{marca} |")
        st.markdown("| horizonte | rio | água na régua |\n|---|---|---|\n"
                    + "\n".join(linhas))

    msg = ss.alerta.avaliar(nivel, aval, ss.rio.chuva_prevista_mm)
    if msg:
        ss.mensagens.insert(0, f"{time.strftime('%H:%M:%S')} · {msg}")
        del ss.mensagens[5:]
    if ss.mensagens:
        st.markdown("#### Alertas enviados")
        for m in ss.mensagens:
            st.caption(m)


def mostrar():
    """Barra lateral da câmera e painel da régua ao vivo."""
    st.sidebar.subheader("Câmera")
    fonte = st.sidebar.radio("Fonte", [REDE, LOCAL])
    if fonte == REDE:
        endereco = st.sidebar.text_input("Endereço da Raspberry",
                                         CR.ENDERECO_PADRAO).rstrip("/")
        caminho_modelo, indice = None, None
    else:
        endereco = None
        caminho_modelo = st.sidebar.text_input("Modelo da régua", modelo_padrao())
        indice = int(st.sidebar.number_input("Índice da câmera", 0, 9,
                                             2 if WINDOWS else 0))
    ligada = st.sidebar.toggle("Câmera ligada", value=True)
    intervalo = st.sidebar.slider("Atualizar a cada (s)", 0.5, 3.0, 1.0, 0.5)
    telegram = st.sidebar.checkbox("Enviar alertas ao Telegram", value=False)

    ss = sessao(telegram)

    st.title("Régua ao vivo")
    st.caption("Leitura real pela câmera. A previsão usa os modelos do rio sobre "
               "um cenário simulado de chuva, e a variação prevista é somada à "
               "régua.")

    b1, b2, _ = st.columns([1, 1, 3])
    if botao(b1, "Previsão de chuva", type="primary"):
        ss.rio.trocar("chuva")
        ss.alerta.ultimo = "_trocou"
    if botao(b2, "Tempo firme"):
        ss.rio.trocar("firme")
        ss.alerta.ultimo = "_trocou"

    # numa feira a Raspberry pode estar desligada ou fora da rede: nesse caso
    # só uma nota discreta no lugar do vídeo, sem erro na tela
    chave_online = f"camera_online:{endereco}"
    if fonte == REDE and chave_online not in ss:
        ss[chave_online] = consultar(endereco) is not None

    @st.fragment(run_every=intervalo if ligada else None)
    def painel_rede():
        d = consultar(endereco)
        online = d is not None
        # o vídeo fica fora do fragmento (recriar o <img> reabriria o stream);
        # quando a Raspberry cai ou volta, recarrega a página pra trocar o
        # vídeo pela nota, ou o contrário
        if online != ss.get(chave_online):
            ss[chave_online] = online
            st.rerun()
        if not online:
            return
        if not d.get("pronto"):
            st.info("A Raspberry está iniciando o modelo...")
            return
        st.caption(f"modelo na Raspberry: {d['ms']:.0f} ms · método: {d['metodo']}")
        mostrar_info(ss, d["nivel_cm"], d["numeros"], d.get("detalhe", ""))

    @st.fragment(run_every=intervalo if ligada else None)
    def painel_local():
        col_img, col_info = st.columns([3, 2], gap="large")
        if not ligada:
            col_img.info("Câmera desligada. Ligue na barra lateral.")
            return

        try:
            modelo = carregar_modelo(caminho_modelo)
        except Exception as e:
            col_img.error(f"Não foi possível carregar o modelo: {e}")
            return

        cap = camera(indice)
        if cap is None:
            col_img.error(f"Não foi possível abrir a câmera {indice}. "
                          "Troque o índice na barra lateral.")
            return

        cap.grab()
        ok, frame = cap.read()
        if not ok:
            col_img.warning("A câmera não devolveu imagem. Tentando de novo...")
            return

        t0 = time.perf_counter()
        res = modelo.predict(frame, imgsz=640, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        numeros, gauges, surfaces = V.extrair(res, modelo.names)
        numeros = V.corrigir_deteccoes(numeros)
        leitura = V.ler_nivel(numeros, gauges, surfaces)
        nivel = ss.filtro.add(leitura.nivel_cm)
        img = V.anotar(frame, res, leitura, numeros, gauges, surfaces)

        with col_img:
            mostrar_imagem(img[:, :, ::-1])
            st.caption(f"inferência: {ms:.0f} ms · método: {leitura.metodo}")

        with col_info:
            mostrar_info(ss, nivel, sorted({d.valor for d in numeros}),
                         leitura.detalhe)

    if fonte == LOCAL:
        painel_local()
    elif ligada:
        col_img, col_info = st.columns([3, 2], gap="large")
        if ss[chave_online]:
            col_img.markdown(CR.html_video(endereco), unsafe_allow_html=True)
        else:
            col_img.caption(f"Câmera sem resposta em {endereco}. Confira se o "
                            "servidor_camera.py está rodando e se os dois estão "
                            "na mesma rede; a imagem volta sozinha.")
        with col_info:
            painel_rede()
    else:
        st.info("Câmera desligada. Ligue na barra lateral.")
