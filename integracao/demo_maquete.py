# -*- coding: utf-8 -*-
"""
demo_maquete.py — HidroVision AI
Demonstração ao vivo: câmera lendo a régua da maquete + cenário simulado do
rio alimentando os modelos XGBoost.

O que é real e o que é simulado nesta demo:

  REAL   a leitura da régua pela câmera (modelo YOLO26 + correção geométrica),
         a tendência calculada do histórico da própria câmera, a projeção de
         quanto falta para os 100 cm e os alertas do módulo de alertas.
  REAL   os três modelos XGBoost — são os mesmos arquivos usados em campo.
  SIMULADO  a série do rio que alimenta os modelos. Numa feira não há enchente
         acontecendo, então a demo injeta um evento de cheia fictício para que
         o modelo responda. Os números que ele devolve são previsão de verdade
         sobre entrada fictícia, não valores escritos à mão.

Os modelos foram treinados na estação 61305000, onde o nível vive entre 14 e
447 cm. A maquete opera em 0 a 100 cm e enche em minutos, então alimentar os
modelos com a escala da maquete daria previsão sem sentido. Por isso as duas
escalas aparecem separadas na tela.

Banco separado (demo_maquete.db) e alertas marcados como DEMO: o monitor real
(monitor.py, banco hidrovision.db) não é tocado.

A água fica parada. Quem dispara o alerta é a previsão: o modelo lê a chuva
prevista na bacia, estima a variação do nível e essa variação é somada à
leitura da régua. Se a subida prevista passar da folga até os 100 cm, o
sistema alerta antes de a água se mexer — que é o comportamento útil em campo.

Uso:
    python demo_maquete.py --webcam 1
    python demo_maquete.py --webcam 1 --auto-chuva 20
    python demo_maquete.py --sem-camera

Teclas: c dispara a previsão de chuva | n volta ao tempo firme
        s salva a tela | r recalcula | q sai
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visao"))

import banco as B
import alertas as A
import pipeline as PL
import preditor as P

DB_DEMO = "demo_maquete.db"
NIVEL_CRITICO = 100.0
LIMIARES_RIO = A.MODOS["estacao"]


class CanalPrefixo:
    """Marca toda mensagem como demonstração, para não confundir com o
    monitor real caso o Telegram esteja ativo."""

    def __init__(self, canal, prefixo="[DEMO maquete] "):
        self.canal = canal
        self.prefixo = prefixo

    def enviar(self, mensagem):
        self.canal.enviar(self.prefixo + mensagem)


def canais_demo(telegram=True):
    canais = [A.CanalConsole(), A.CanalArquivo("alertas_demo.log")]
    if telegram and os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        try:
            canais.append(A.CanalTelegram())
            print("alertas: canal do Telegram ativo (mensagens marcadas como DEMO)")
        except Exception as e:
            print(f"[aviso] Telegram não configurado ({e})")
    else:
        print("alertas: Telegram inativo nesta demo")
    return [CanalPrefixo(c) for c in canais]


class RioSimulado:
    """Série horária fictícia do rio, em dois cenários.

    tempo firme: rio estável, sem chuva — o modelo corretamente prevê que
                 nada acontece, e o sistema fica em silêncio.
    chuva:       frente de chuva sobre a bacia alta e rio já subindo — é a
                 entrada que faz o modelo projetar a cheia.

    O que o modelo devolve é previsão de verdade; fictícia é só a série que
    entra, porque numa feira não há enchente acontecendo.
    """

    CENARIOS = {
        "firme": {"nivel_agora": 170.0, "base": 165.0, "horas_evento": 6,
                  "chuva_pico": 0.0, "chuva_prevista_mm": 0},
        "chuva": {"nivel_agora": 300.0, "base": 140.0, "horas_evento": 4,
                  "chuva_pico": 15.0, "chuva_prevista_mm": 62},
    }

    def __init__(self, pasta_modelos, cenario="firme", dias=30):
        self.dias = dias
        self.cenario = cenario
        try:
            self.preditor = P.Preditor(pasta_modelos)
        except FileNotFoundError as e:
            print(f"[aviso] {e} — demo sem previsão do rio")
            self.preditor = None
        self.previsoes = None

    @property
    def cfg(self):
        return self.CENARIOS[self.cenario]

    @property
    def chuva_prevista_mm(self):
        return self.cfg["chuva_prevista_mm"]

    def trocar(self, cenario):
        self.cenario = cenario
        return self.prever()

    def serie(self):
        c = self.cfg
        horas = self.dias * 24
        fim = pd.Timestamp.now(tz=B.FUSO).floor("h")
        idx = pd.date_range(end=fim, periods=horas, freq="h")
        nivel = np.full(horas, c["base"], dtype=float)
        chuva = np.zeros(horas)
        for k in range(c["horas_evento"]):
            i = horas - c["horas_evento"] + k
            avanco = ((k + 1) / c["horas_evento"]) ** 1.8
            nivel[i] = c["base"] + (c["nivel_agora"] - c["base"]) * avanco
        if c["chuva_pico"] > 0:
            for k in range(c["horas_evento"] + 10):
                i = horas - c["horas_evento"] - 10 + k
                chuva[i] = c["chuva_pico"] * np.exp(-((k - 10) ** 2) / 40)
        return pd.DataFrame({"nivel_cm": nivel, "chuva_mm": chuva}, index=idx)

    def prever(self):
        if self.preditor is None:
            return None
        self.previsoes = self.preditor.prever(self.serie())
        return self.previsoes

    def variacoes(self):
        """Variação prevista em cada horizonte — é o que transfere para a
        régua local, já que os modelos preveem delta e não nível absoluto."""
        if not self.previsoes:
            return {}
        atual = self.previsoes["nivel_atual"]
        return {h: self.previsoes[f"{h}h"] - atual
                for h in (6, 12, 24) if f"{h}h" in self.previsoes}


def avaliar_previsao_local(nivel_regua, variacoes, critico=NIVEL_CRITICO):
    """Soma a variação prevista do rio à leitura da régua e diz em que
    horizonte a água alcançaria a área urbana."""
    if nivel_regua is None or not variacoes:
        return None
    projecao = {h: nivel_regua + d for h, d in variacoes.items()}
    cruza = next((h for h in sorted(projecao) if projecao[h] >= critico), None)
    urgencia = {6: "emergencia", 12: "alerta", 24: "atencao"}.get(cruza)
    if urgencia is None and projecao and max(projecao.values()) >= critico - 15:
        urgencia = "atencao"
    return {"projecao": projecao, "cruza_em_h": cruza, "urgencia": urgencia,
            "folga_cm": critico - nivel_regua}


class AlertaPrevisao:
    """Dispara o alerta de previsão uma vez por mudança de patamar, para não
    repetir a mesma mensagem a cada frame."""

    def __init__(self, canais):
        self.canais = canais
        self.ultimo = None

    def avaliar(self, nivel_regua, aval, chuva_mm):
        if aval is None:
            return None
        urgencia = aval["urgencia"]
        if urgencia == self.ultimo:
            return None
        self.ultimo = urgencia
        if urgencia is None:
            msg = ("[PREVISÃO] Normalizado: sem chuva relevante prevista, "
                   f"água em {nivel_regua:.0f} cm sem risco de transbordamento "
                   "nas próximas 24 h")
        else:
            proj = aval["projecao"]
            partes = ", ".join(f"{h} h: {proj[h]:.0f} cm" for h in sorted(proj))
            quando = (f"transborda em até {aval['cruza_em_h']} h"
                      if aval["cruza_em_h"] else "chega perto do limite em 24 h")
            msg = (f"[PREVISÃO] {A.NOMES[urgencia]}: água em {nivel_regua:.0f} cm "
                   f"(folga de {aval['folga_cm']:.0f} cm) e {chuva_mm} mm de chuva "
                   f"previstos na bacia — modelo projeta {partes}; {quando}")
        for c in self.canais:
            c.enviar(msg)
        return msg


def linhas_painel(estado, rio, aval):
    linhas = []
    ult = estado["ultima_leitura"]
    nivel = ult["nivel_cm"] if ult is not None else None

    linhas.append(("REGUA  (camera, leitura real)", "titulo"))
    if nivel is None:
        linhas.append(("aguardando leitura", "aviso"))
    else:
        linhas.append((f"nivel: {nivel:.0f} cm", "forte"))
        linhas.append((f"folga ate transbordar: {NIVEL_CRITICO - nivel:.0f} cm", "normal"))

    linhas.append(("", "normal"))
    linhas.append(("PREVISAO DO RIO  (cenario simulado)", "titulo"))
    if rio.previsoes is None:
        linhas.append(("sem previsao", "aviso"))
    else:
        if rio.chuva_prevista_mm:
            linhas.append((f"chuva na bacia: {rio.chuva_prevista_mm} mm em 12 h", "normal"))
        else:
            linhas.append(("tempo firme, sem chuva prevista", "normal"))
        for h, d in sorted(rio.variacoes().items()):
            linhas.append((f"  {h:2d} h: {d:+.0f} cm no rio", "normal"))

    if aval is not None and nivel is not None:
        linhas.append(("", "normal"))
        linhas.append(("AGUA NA REGUA (projetada)", "titulo"))
        for h, v in sorted(aval["projecao"].items()):
            marca = "  TRANSBORDA" if v >= NIVEL_CRITICO else ""
            tipo = "aviso" if v >= NIVEL_CRITICO else "normal"
            linhas.append((f"  {h:2d} h: {v:.0f} cm{marca}", tipo))
        if aval["urgencia"]:
            linhas.append((f">> {A.NOMES[aval['urgencia']]}", "alarme"))
        else:
            linhas.append((">> situacao normal", "normal"))
    return linhas


def desenhar_painel(img, linhas, cv2):
    largura = 330
    h, w = img.shape[:2]
    painel = np.zeros((h, largura, 3), dtype=img.dtype)
    painel[:] = (28, 24, 20)

    CORES = {"titulo": (180, 220, 255), "forte": (245, 245, 245),
             "normal": (200, 200, 200), "aviso": (0, 170, 255),
             "alarme": (60, 60, 235)}
    y = 34
    for texto, tipo in linhas:
        if texto:
            escala = 0.52 if tipo == "titulo" else 0.58 if tipo in ("forte", "alarme") else 0.48
            grossura = 2 if tipo in ("titulo", "forte", "alarme") else 1
            cv2.putText(painel, texto, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        escala, CORES[tipo], grossura, cv2.LINE_AA)
        y += 26 if texto else 12
    cv2.putText(painel, "DEMO - dados do rio simulados", (14, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
    return np.hstack([img, painel])


def ciclo_demo(p, nivel_cm, metodo, confianca, menor, ts=None):
    return p.processar_leitura(nivel_cm, metodo=metodo, confianca=confianca,
                               menor_numero=menor, ts=ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=os.path.join("..", "visao", "hidrovision_v06_regua.pt"))
    ap.add_argument("--modelos", default=os.path.join("..", "preditivo", "modelos"))
    ap.add_argument("--webcam", type=int, default=2 if sys.platform.startswith("win") else 0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--intervalo", type=float, default=3.0,
                    help="segundos entre gravações no banco")
    ap.add_argument("--auto-chuva", type=float, default=0.0,
                    help="segundos até a previsão de chuva entrar sozinha (0 = só na tecla c)")
    ap.add_argument("--db", default=DB_DEMO)
    ap.add_argument("--manter-banco", action="store_true")
    ap.add_argument("--sem-telegram", action="store_true")
    ap.add_argument("--sem-camera", action="store_true",
                    help="ensaia sem a maquete, com a régua fixa em 40 cm")
    ap.add_argument("--nivel-ensaio", type=float, default=40.0)
    args = ap.parse_args()

    if not args.manter_banco and os.path.exists(args.db):
        os.remove(args.db)

    canais = canais_demo(not args.sem_telegram)
    rio = RioSimulado(args.modelos)
    rio.prever()
    alerta = AlertaPrevisao(canais)

    p = PL.Pipeline(modo="maquete", db=args.db, pasta_modelos=args.modelos,
                    canais=canais)
    p.preditor = None

    if args.sem_camera:
        return ensaio_sem_camera(p, rio, alerta, args)

    import cv2
    import mdYOLO as V
    from ultralytics import YOLO

    modelo = YOLO(args.modelo)
    V.montar_paleta(modelo.names)
    print(f"modelo: {args.modelo} | classes: {len(modelo.names)}")

    cap = V.abrir_camera(args.webcam)
    if cap is None:
        print("não foi possível abrir a câmera", args.webcam)
        return

    camera = V.CameraAoVivo(cap)
    detector = V.DetectorAoVivo(modelo, camera, args.imgsz)
    ultimo = 0.0
    inicio = time.time()
    chuva_entrou = False
    estado = p.estado_atual()
    aval = None
    print("c previsão de chuva | n tempo firme | s salvar tela | r recalcular | q sair")
    while True:
        frame = camera.ler()
        if frame is None:
            time.sleep(0.01)
            continue
        r = detector.ler()

        agora = time.time()
        if args.auto_chuva and not chuva_entrou and agora - inicio >= args.auto_chuva:
            chuva_entrou = True
            rio.trocar("chuva")
            print("  previsão de chuva entrou no cenário")

        suave = r["suave"] if r else None
        if suave is not None and agora - ultimo >= args.intervalo:
            ultimo = agora
            leitura, numeros = r["leitura"], r["numeros"]
            menor = min((d.valor for d in numeros), default=None)
            ciclo_demo(p, suave, leitura.metodo, leitura.confianca, menor, None)
            estado = p.estado_atual()
            aval = avaliar_previsao_local(suave, rio.variacoes())
            alerta.avaliar(suave, aval, rio.chuva_prevista_mm)

        img = V.quadro_anotado(frame, r)
        img = desenhar_painel(img, linhas_painel(estado, rio, aval), cv2)
        cv2.imshow("HidroVision - DEMO maquete", img)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("c"):
            rio.trocar("chuva")
            alerta.ultimo = None
        if tecla == ord("n"):
            rio.trocar("firme")
            alerta.ultimo = None
        if tecla == ord("s"):
            nome = f"demo_{pd.Timestamp.now():%H%M%S}.png"
            cv2.imwrite(nome, img)
            print("  tela salva em", nome)
        if tecla == ord("r"):
            rio.prever()
    detector.parar()
    camera.parar()
    cv2.destroyAllWindows()


def ensaio_sem_camera(p, rio, alerta, args):
    """Ensaio sem maquete: régua parada no nível informado, a chuva entra no
    meio do caminho e o alerta vem da previsão."""
    nivel = args.nivel_ensaio
    print(f"\nensaio sem câmera — régua parada em {nivel:.0f} cm\n")
    for etapa, cenario in (("tempo firme", "firme"), ("previsão de chuva", "chuva")):
        rio.trocar(cenario)
        ciclo_demo(p, nivel, "ensaio", 0.9, int(nivel // 10 * 10), None)
        estado = p.estado_atual()
        aval = avaliar_previsao_local(nivel, rio.variacoes())
        print(f"--- {etapa} ---")
        for texto, _ in linhas_painel(estado, rio, aval):
            if texto:
                print("   ", texto)
        alerta.avaliar(nivel, aval, rio.chuva_prevista_mm)
        print()
        time.sleep(args.intervalo)


if __name__ == "__main__":
    main()
