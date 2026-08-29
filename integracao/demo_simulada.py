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
    python demo_simulada.py --modo estacao --replay-ana dados_treino.csv --rapido
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

    print(f"\n{'-'*72}\nAlertas gravados:")
    al = p.banco.alertas_recentes(10)
    if al.empty:
        print("  nenhum")
    else:
        for _, a in al.iloc[::-1].iterrows():
            print(f"  {a.ts:%H:%M}  {a.tipo:11s} {a.estado:10s} "
                  f"{a.nivel_cm:5.0f} cm")


def _valor_previsao(prev, chave):
    """Extrai um horizonte da previsão, tolerante ao formato do dicionário."""
    if not prev:
        return None
    for k in (chave, f"{chave}h", f"t+{chave}", f"t+{chave}h"):
        v = prev.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def demo_replay_ana(p, csv, rapido=False, pausa=0.25):
    """
    Reproduz a cheia real de março/2026 da estação 61305000, hora a hora.

    A cada hora imprime o nível observado, a tendência calculada e o nível que
    os modelos XGBoost projetam para 6, 12 e 24 horas à frente. Os alertas
    aparecem no meio da tabela, no instante em que disparam.
    """
    print(f"\n{'='*78}")
    print("REPLAY — cheia real de março/2026 (estação 61305000, dados da ANA)")
    print(f"{'='*78}")
    print("Leitura de 1 cm: são os dados oficiais da estação, não a câmera.")
    print("Limiares desta estação: atenção 228 | alerta 304 | emergência 388 cm.")
    print("As colunas de previsão trazem o nível projetado pelos modelos.\n")

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

    print(f"  {'data/hora':>13} {'nível':>9} {'tendência':>16} "
          f"{'prev 6h':>10} {'prev 12h':>10} {'prev 24h':>10}")
    print(f"  {'-'*13} {'-'*9} {'-'*16} {'-'*10} {'-'*10} {'-'*10}")

    trajetoria, nivel_cruzou = {}, {}
    pico_ts, pico_nivel = None, -1e9

    for _, row in trecho.iterrows():
        ts, nivel = row["datahora"], float(row["nivel_cm"])
        r = p.processar_leitura(nivel, "replay", 1.0, ts=ts)

        if nivel > pico_nivel:
            pico_ts, pico_nivel = ts, nivel

        if r.tendencia is not None and r.tendencia.taxa_cm_h is not None:
            seta = getattr(r.tendencia, "seta", "")
            tend = f"{r.tendencia.taxa_cm_h:+.1f} cm/h {seta}"
        else:
            tend = r.tendencia.rotulo if r.tendencia is not None else "--"

        prev = getattr(r, "previsoes", None)
        cols = []
        for h in ("6", "12", "24"):
            v = _valor_previsao(prev, h)
            cols.append(f"{v:6.0f} cm" if v is not None else "      —")

        print(f"  {ts:%d/%m %H:%M} {nivel:6.0f} cm {tend:>16} "
              f"{cols[0]:>10} {cols[1]:>10} {cols[2]:>10}")

        # os alertas do ciclo aparecem logo abaixo da linha que os gerou
        for ev in r.eventos_alerta:
            marca = ">>" if ev.estado in ("disparado", "agravado") else "--"
            print(f"      {marca} {ev.mensagem}")
            if ev.estado in ("disparado", "agravado"):
                if ev.categoria == "trajetoria" and ev.tipo not in trajetoria:
                    trajetoria[ev.tipo] = (ts, ev.valor)
                if ev.categoria == "nivel" and ev.tipo not in nivel_cruzou:
                    nivel_cruzou[ev.tipo] = (ts, nivel)

        if not rapido:
            time.sleep(pausa)

    # ------------------------------------------------------------------
    print(f"\n{'-'*78}")
    print(f"{len(trecho)} horas processadas. "
          f"Pico do evento: {pico_nivel:.0f} cm em {pico_ts:%d/%m %H:%M}.")

    print(f"\n{'-'*78}")
    print("Antecedência: alerta de TRAJETÓRIA vs. o nível cruzando o limiar")
    houve = False
    for tipo in ("atencao", "alerta", "emergencia"):
        if tipo in trajetoria and tipo in nivel_cruzou:
            t_traj, v_traj = trajetoria[tipo]
            t_real, v_real = nivel_cruzou[tipo]
            horas = (t_real - t_traj).total_seconds() / 3600
            print(f"  {tipo:11s} trajetória avisou {t_traj:%d/%m %H:%M} "
                  f"({v_traj:.0f} cm) | nível cruzou {t_real:%d/%m %H:%M} "
                  f"({v_real:.0f} cm) -> {horas:+.0f} h de antecedência")
            houve = True
        elif tipo in nivel_cruzou:
            t_real, v_real = nivel_cruzou[tipo]
            print(f"  {tipo:11s} nível cruzou {t_real:%d/%m %H:%M} "
                  f"({v_real:.0f} cm), sem aviso prévio de trajetória")
            houve = True
    if not houve:
        print("  nenhum alerta disparado no período")
    print("\n  Nesta estação o nível crítico é 447 cm e o rio subiu devagar, então")
    print("  o tempo projetado ficou acima do horizonte de 24 h e a trajetória")
    print("  não entrou em faixa de urgência. É o comportamento esperado: a")
    print("  extrapolação só alerta quando a subida é rápida o bastante.")

    print(f"\n{'-'*78}\nAlertas gravados no período:")
    al = p.banco.alertas_recentes(20)
    if al.empty:
        print("  nenhum")
    else:
        for _, a in al.iloc[::-1].iterrows():
            print(f"  {a.ts:%d/%m %H:%M}  {a.tipo:11s} {a.estado:11s} "
                  f"{a.nivel_cm:6.0f} cm")


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
    ap.add_argument("--rapido", action="store_true",
                    help="sem pausa entre as linhas")
    ap.add_argument("--pausa", type=float, default=0.25,
                    help="segundos entre as linhas do replay (padrão 0,25)")
    args = ap.parse_args()

    if os.path.exists(args.db):
        os.remove(args.db)
    p = PL.Pipeline(modo=args.modo, db=args.db, pasta_modelos=args.modelos,
                    prever_a_cada_min=0)
    if args.replay_ana:
        demo_replay_ana(p, args.replay_ana, rapido=args.rapido,
                        pausa=args.pausa)
    else:
        demo_maquete(p, rapido=args.rapido, interpolada=args.interpolada)