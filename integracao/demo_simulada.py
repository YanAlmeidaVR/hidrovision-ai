# -*- coding: utf-8 -*-
"""
demo_simulada.py — HidroVision AI (Fase 3)
Demonstração do pipeline completo SEM câmera e SEM água.

O que esta demo simula, e por que assim:

A câmera não mede centímetros — ela vê quais números da régua ainda estão
visíveis. Com marcações de 10 em 10 cm, quando o "40" desaparece sob a água,
a única informação certa é "o nível passou de 40". A leitura, portanto, é um
DEGRAU de 10 cm, não um valor contínuo.

Por isso a demo (modo padrão) reproduz esse comportamento: a água sobe
suavemente por dentro, mas o sistema recebe apenas o menor número visível.
É o cenário conservador e realista.

    python demo_simulada.py                      # leitura em degraus (realista)
    python demo_simulada.py --interpolada        # leitura fina (se a
                                                 # geometria funcionar bem)
    python demo_simulada.py --modo estacao --replay-ana dados_treino.csv
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import banco as B
import pipeline as PL

PASSO = 10.0          # cm entre marcações da régua


def leitura_da_camera(nivel_real, interpolada=False, passo=PASSO):
    """
    Converte o nível real (que só a água sabe) na leitura que a câmera produz.

    interpolada=False : menor número visível — degraus de `passo` cm.
    interpolada=True  : valor refinado pela geometria, com ruído de ~1,5 cm.
    """
    if interpolada:
        return round(nivel_real + np.random.uniform(-1.5, 1.5), 1)
    return float(int(nivel_real // passo) * passo)


def demo_maquete(p, rapido=False, interpolada=False):
    # Os modelos XGBoost foram treinados na estação 61305000, onde o nível vive
    # entre 14 e 447 cm e o rio responde em horas. A maquete opera em 0-100 cm e
    # enche em minutos: alimentar os modelos com essa escala produz previsão sem
    # sentido (prevê estiagem enquanto a água sobe). Na maquete, portanto, o
    # sistema demonstra leitura + tendência + alerta; a previsão é demonstrada
    # com os dados reais da estação (--replay-ana).
    p.preditor = None
    modo_txt = "INTERPOLADA (~2-3 cm)" if interpolada else "EM DEGRAUS (10 cm)"
    print(f"\n{'='*72}")
    print(f"DEMO MAQUETE — régua urbana de 1 m | leitura {modo_txt}")
    print(f"{'='*72}")
    print("A régua está junto à área urbana: quando a água chega aos 100 cm,")
    print("ela atinge a cidade. Quanto mais alto o número visível, menor a folga.")
    if not interpolada:
        print("A câmera informa o menor número visível — a leitura só muda quando")
        print("a água cobre a marcação seguinte, em degraus de 10 cm.")
    print()

    # semeia 30 h de histórico (o preditor usa lags de até 24 h)
    t0 = pd.Timestamp.now(tz=B.FUSO).floor("h") - pd.Timedelta(hours=30)
    for i in range(30):
        # semeia no mesmo valor da primeira leitura, senão a descontinuidade
        # produz uma tendência negativa espúria nos primeiros ciclos
        p.banco.gravar_leitura(30.0, "seed", ts=t0 + pd.Timedelta(hours=i),
                               janela_mediana=1)
    p._ultima_previsao_ts = None

    # a água sobe ~14 cm/h; a câmera lê a cada 5 min
    t1 = t0 + pd.Timedelta(hours=30)
    print(f"  {'hora':>5} {'real':>6} {'lido':>6} {'faixa':>13} "
          f"{'tendência':>14} {'atinge 100 cm em':>17}")
    print(f"  {'-'*5} {'-'*6} {'-'*6} {'-'*13} {'-'*14} {'-'*17}")

    for passo_i in range(25):                    # 25 leituras = ~2 h
        minutos = passo_i * 5
        nivel_real = 38 + 14 * (minutos / 60)    # sobe 14 cm/h
        if nivel_real > 96:
            break
        lido = leitura_da_camera(nivel_real, interpolada)
        ts = t1 + pd.Timedelta(minutes=minutos)

        r = p.processar_leitura(lido, "interpolacao" if interpolada
                                else "menor_numero", 0.8, ts=ts)

        # faixa: o que o sistema pode afirmar com certeza
        if interpolada:
            faixa = f"~{r.nivel_gravado:.0f} cm"
        else:
            base = r.nivel_gravado
            faixa = f"{base:.0f}-{base+PASSO:.0f} cm"

        if r.tendencia and r.tendencia.taxa_cm_h is not None:
            tend = f"{r.tendencia.taxa_cm_h:+.0f} cm/h {r.tendencia.seta}"
        else:
            tend = r.tendencia.rotulo if r.tendencia else "--"
        proj_txt = "—"
        if r.projecao is not None:
            if r.projecao.estado == "subindo":
                proj_txt = r.projecao.tempo_formatado
            elif r.projecao.estado == "descendo":
                proj_txt = "recuando"
            elif r.projecao.estado == "estavel":
                proj_txt = "estável"
        print(f"  {ts:%H:%M} {nivel_real:6.1f} {lido:6.1f} {faixa:>13} "
              f"{tend:>14} {proj_txt:>17}")
        if not rapido:
            time.sleep(0.35)

    print(f"\n{'-'*72}\nO que o dashboard mostraria agora:")
    e = p.estado_atual()
    ult = e["ultima_leitura"]["nivel_cm"]
    if interpolada:
        print(f"  nível: ~{ult:.0f} cm")
    else:
        print(f"  nível: entre {ult:.0f} e {ult+PASSO:.0f} cm "
              f"(menor número visível: {ult:.0f})")
    t = e["tendencia"]
    print(f"  tendência: {t.rotulo} "
          f"({t.taxa_cm_h:+.1f} cm/h)" if t.taxa_cm_h is not None
          else f"  tendência: {t.rotulo}")
    print(f"  resolução da leitura: {t.resolucao} — {t.detalhe}")
    print(f"  modo: {e['modo']} | limiares: {e['limiares']}")
    pj = e.get("projecao")
    if pj is not None:
        print(f"  {pj.resumo()}")
        if pj.detalhe:
            print(f"  ({pj.detalhe})")

    print(f"\n{'-'*72}")
    print("Como a projeção é calculada:")
    print("  A régua fica junto à área urbana: 100 cm é o ponto em que a água")
    print("  atinge a cidade. A projeção é uma extrapolação da tendência —")
    print("  folga em cm dividida pela taxa em cm/h. Ela só se aplica enquanto o")
    print("  nível sobe; se estabiliza ou recua, é cancelada.")
    print("  Os modelos XGBoost não entram aqui: foram treinados na cota do rio")
    print("  na estação 61305000 (14 a 447 cm), outra escala e outra física.")
    print("  Eles são demonstrados com os dados reais da estação:")
    print("      python demo_simulada.py --modo estacao --replay-ana dados_treino.csv")

    print(f"\n{'-'*72}\nAlertas gravados:")
    al = p.banco.alertas_recentes(10)
    if al.empty:
        print("  nenhum")
    else:
        for _, a in al.iloc[::-1].iterrows():
            print(f"  {a.ts:%H:%M}  {a.tipo:11s} {a.estado:10s} "
                  f"{a.nivel_cm:5.0f} cm  [{a.origem}]")


def demo_replay_ana(p, csv):
    """Reproduz a cheia real de março/2026 da estação 61305000, hora a hora."""
    print(f"\n{'='*72}")
    print("REPLAY — cheia real de março/2026 (estação 61305000, dados da ANA)")
    print(f"{'='*72}")
    print("Aqui a leitura é de 1 cm: são os dados oficiais da estação, não a")
    print("câmera. Serve para verificar o alerta antecipado em evento real.\n")

    df = pd.read_csv(csv, parse_dates=["datahora"])
    df["datahora"] = pd.to_datetime(df["datahora"], utc=True).dt.tz_convert(B.FUSO)
    ini = pd.Timestamp("2026-03-09", tz=B.FUSO)
    fim = pd.Timestamp("2026-03-13", tz=B.FUSO)
    trecho = df[(df["datahora"] >= ini) & (df["datahora"] <= fim)].dropna(
        subset=["nivel_cm"])
    hist = df[(df["datahora"] < ini)
              & (df["datahora"] >= ini - pd.Timedelta(days=3))].dropna(
        subset=["nivel_cm"])
    for _, row in hist.iterrows():
        p.banco.gravar_leitura(row["nivel_cm"], "replay",
                               ts=row["datahora"], janela_mediana=1)
    p._ultima_previsao_ts = None

    primeiro_alerta, cruzou_de_fato = {}, {}
    for _, row in trecho.iterrows():
        r = p.processar_leitura(row["nivel_cm"], "replay", 1.0,
                               ts=row["datahora"])
        for estado, tipo, origem, valor in r.eventos_alerta:
            if estado != "disparado":
                continue
            if origem.startswith("previsao") and tipo not in primeiro_alerta:
                primeiro_alerta[tipo] = (row["datahora"], origem, valor)
            if origem == "nivel_atual" and tipo not in cruzou_de_fato:
                cruzou_de_fato[tipo] = (row["datahora"], row["nivel_cm"])

    print(f"{len(trecho)} horas processadas.\n")
    print("Antecedência do alerta (previsão vs. nível real cruzando o limiar):")
    for tipo in ("atencao", "alerta", "emergencia"):
        if tipo in primeiro_alerta and tipo in cruzou_de_fato:
            t_prev, origem, v_prev = primeiro_alerta[tipo]
            t_real, v_real = cruzou_de_fato[tipo]
            horas = (t_real - t_prev).total_seconds() / 3600
            print(f"  {tipo:11s} avisado {t_prev:%d/%m %H:%M} ({origem}, "
                  f"{v_prev:.0f} cm) | cruzou {t_real:%d/%m %H:%M} "
                  f"({v_real:.0f} cm) -> {horas:+.0f} h de antecedência")
        elif tipo in cruzou_de_fato:
            t_real, v_real = cruzou_de_fato[tipo]
            print(f"  {tipo:11s} cruzou {t_real:%d/%m %H:%M} sem aviso prévio")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", default="maquete", choices=["maquete", "estacao"])
    ap.add_argument("--db", default="demo.db")
    ap.add_argument("--modelos", default=".",
                    help="pasta com os modelo_delta_*.json")
    ap.add_argument("--replay-ana", metavar="CSV",
                    help="reproduz a cheia real de mar/2026 desse CSV")
    ap.add_argument("--interpolada", action="store_true",
                    help="simula leitura fina em vez de degraus de 10 cm")
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.db):
        os.remove(args.db)
    p = PL.Pipeline(modo=args.modo, db=args.db, pasta_modelos=args.modelos,
                    prever_a_cada_min=0)
    if args.replay_ana:
        demo_replay_ana(p, args.replay_ana)
    else:
        demo_maquete(p, rapido=args.rapido, interpolada=args.interpolada)
