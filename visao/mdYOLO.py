# -*- coding: utf-8 -*-
"""
leitura_nivel.py — HidroVision AI
Lê o nível da água na régua linimétrica a partir do modelo YOLO26 (V05).

Método (na ordem em que é tentado):
  1) INTERPOLAÇÃO — com dois ou mais números detectados obtém-se a escala
     px/cm; a borda inferior do menor número visível marca a linha d'água.
     Resolução esperada: ~2-3 cm.
  2) SURFACE — se a classe `surface` for detectada com confiança alta, usa a
     posição dela para refinar. (No V05 a surface tem recall 0,39, por isso
     é auxiliar e não principal.)
  3) MENOR NÚMERO — fallback: o menor número visível define a faixa.
     Resolução ~10 cm.

Uso:
    python leitura_nivel.py --modelo hidrovision_v05.pt --imagem foto.jpg
    python leitura_nivel.py --modelo hidrovision_v05.pt --pasta ./fotos --csv saida.csv
    python leitura_nivel.py --modelo hidrovision_v05.pt --webcam 0
"""
import argparse
import csv
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import geometria as G
    GEOMETRIA = True
except ImportError:      # roda sem correção geométrica se o módulo faltar
    GEOMETRIA = False
    print("aviso: geometria.py não encontrado — correção geométrica desativada")

# classes numéricas do modelo (as demais são gauge e surface)
NUMERICAS = {"0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"}
CONF_MIN = 0.35          # confiança mínima para considerar uma detecção
CONF_SURFACE = 0.60      # surface só entra se vier bem confiante
JANELA_MEDIANA = 5       # leituras para o filtro de mediana


@dataclass
class Deteccao:
    classe: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    @property
    def valor(self):
        return int(self.classe)


@dataclass
class Leitura:
    nivel_cm: float | None
    metodo: str
    confianca: float
    n_numeros: int
    detalhe: str = ""
    correcoes: list = field(default_factory=list)
    lacunas: list = field(default_factory=list)


def extrair(resultado, nomes):
    """Converte o resultado do YOLO em listas de detecções por tipo."""
    numeros, gauges, surfaces = [], [], []
    if resultado.boxes is None:
        return numeros, gauges, surfaces
    for b in resultado.boxes:
        conf = float(b.conf[0])
        if conf < CONF_MIN:
            continue
        classe = nomes[int(b.cls[0])]
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        det = Deteccao(classe, conf, x1, y1, x2, y2)
        if classe in NUMERICAS:
            numeros.append(det)
        elif classe == "gauge":
            gauges.append(det)
        elif classe == "surface":
            surfaces.append(det)
    return numeros, gauges, surfaces


def escala_px_por_cm(numeros):
    """
    Estima px/cm por regressão linear entre valor (cm) e posição vertical (px).
    Usa todos os números detectados — mais robusto que só o par mais próximo.
    Retorna (px_por_cm, r2) ou (None, 0) se não der.
    """
    if len(numeros) < 2:
        return None, 0.0
    valores = np.array([n.valor for n in numeros], dtype=float)
    ys = np.array([n.cy for n in numeros], dtype=float)
    if len(np.unique(valores)) < 2:
        return None, 0.0
    # y = a*valor + b ; a é negativo (valor maior fica mais alto = y menor)
    a, b = np.polyfit(valores, ys, 1)
    pred = a * valores + b
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    px_por_cm = abs(a)
    if px_por_cm < 0.1:                     # escala absurda
        return None, r2
    return px_por_cm, r2


def ler_nivel(numeros, gauges, surfaces):
    """Aplica a cascata de métodos e devolve a melhor leitura possível."""
    if not numeros:
        if gauges:
            return Leitura(None, "sem_numeros", 0.0, 0,
                           "régua detectada mas nenhum número visível")
        return Leitura(None, "sem_regua", 0.0, 0, "nada detectado")

    correcoes, lacunas = [], []
    ajuste = None
    if GEOMETRIA and len(numeros) >= 2:
        numeros, ajuste, correcoes = G.corrigir(numeros)
        lacunas = G.detectar_lacunas(numeros, ajuste)

    numeros = sorted(numeros, key=lambda d: d.valor)
    menor = numeros[0]
    conf_media = float(np.mean([n.conf for n in numeros]))

    if ajuste is not None:
        px_cm, r2 = ajuste.px_por_cm, ajuste.r2
    else:
        px_cm, r2 = escala_px_por_cm(numeros)

    extras = []
    if correcoes:
        extras.append(f"{len(correcoes)} correção(ões) geométrica(s)")
    if lacunas:
        extras.append(f"não detectados: {lacunas}")
    sufixo = (" | " + "; ".join(extras)) if extras else ""

    # --- método 2: surface confiável + escala ---
    if px_cm and surfaces:
        s = max(surfaces, key=lambda d: d.conf)
        if s.conf >= CONF_SURFACE:
            # y cresce para baixo: superfície abaixo do número => nível menor
            nivel = menor.valor - (s.cy - menor.cy) / px_cm
            nivel = float(np.clip(nivel, 0, 100))
            return Leitura(nivel, "surface", min(conf_media, s.conf),
                           len(numeros), f"px/cm={px_cm:.2f} r2={r2:.3f}{sufixo}",
                           correcoes, lacunas)

    # --- método 1: interpolação pela borda do menor número ---
    if px_cm and r2 > 0.90:
        # a borda inferior da bbox do menor número visível ~ linha d'água
        nivel = menor.valor - (menor.y2 - menor.cy) / px_cm
        nivel = float(np.clip(nivel, 0, 100))
        metodo = "geometria" if correcoes or lacunas else "interpolacao"
        return Leitura(nivel, metodo, conf_media, len(numeros),
                       f"px/cm={px_cm:.2f} r2={r2:.3f}{sufixo}",
                       correcoes, lacunas)

    # --- método 3: fallback pelo menor número ---
    return Leitura(float(menor.valor), "menor_numero", menor.conf, len(numeros),
                   f"escala indisponível — resolução ±10 cm{sufixo}",
                   correcoes, lacunas)


class FiltroMediana:
    """Suaviza ruído de leitura antes de gravar/alertar."""

    def __init__(self, n=JANELA_MEDIANA):
        self.n = n
        self.buf: list[float] = field(default_factory=list)
        self.buf = []

    def add(self, valor):
        if valor is None:
            return None
        self.buf.append(valor)
        if len(self.buf) > self.n:
            self.buf.pop(0)
        return float(np.median(self.buf))


def faixa_do_nivel(leitura, numeros=None):
    """
    Converte a leitura numa FAIXA de 10 cm, que é a resolução que o sistema
    garante hoje. A estimativa interpolada aparece como valor aproximado, não
    como medida validada — a acurácia em centímetros ainda não foi medida em
    campo com gabarito.
    """
    if leitura.nivel_cm is None:
        return None
    base = int(leitura.nivel_cm // 10) * 10
    return base, min(base + 10, 100)


def anotar(frame, resultado, leitura, numeros=None):
    """
    Desenha as detecções e uma faixa inferior discreta informando o menor
    número visível e o nível aproximado. Sem painel sobre a imagem — as caixas
    do detector ficam livres.
    """
    img = resultado.plot()
    h, w = img.shape[:2]

    VERDE = (60, 200, 90)
    AMBAR = (0, 170, 255)
    VERM = (60, 60, 235)
    BRANCO = (245, 245, 245)

    if leitura.nivel_cm is None:
        if leitura.metodo == "sem_numeros":
            texto, cor = "regua detectada - nenhum numero visivel", AMBAR
        else:
            texto, cor = "procurando regua", VERM
    else:
        menor = int(min(int(d.classe) for d in numeros)) if numeros else None
        cor = (VERDE if leitura.metodo in ("interpolacao", "geometria", "surface")
               else AMBAR)
        partes = []
        if menor is not None:
            partes.append(f"menor numero visivel: {menor}")
        partes.append(f"nivel aproximado: ~{leitura.nivel_cm:.0f} cm")
        texto = "   |   ".join(partes)

    # faixa inferior semitransparente
    alt = 40
    y0 = h - alt
    sub = img[y0:h, 0:w].copy()
    img[y0:h, 0:w] = cv2.addWeighted(sub, 0.25, np.zeros_like(sub), 0.75, 0)
    cv2.rectangle(img, (0, y0), (w, y0 + 3), cor, -1)
    cv2.putText(img, texto, (14, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                cor if leitura.nivel_cm is None else BRANCO, 2)

    # avisos da geometria, à direita da faixa
    aviso = []
    if leitura.correcoes:
        aviso.append(f"{len(leitura.correcoes)} corrigido(s)")
    if leitura.lacunas:
        aviso.append(f"nao detectados: {leitura.lacunas}")
    if aviso:
        msg = " | ".join(aviso)
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, msg, (max(14, w - tw - 14), h - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, AMBAR, 1)

    # barra vertical da régua na lateral direita
    if leitura.nivel_cm is not None:
        bx, bw = w - 40, 20
        by, bh = 14, min(280, h - alt - 28)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (90, 90, 90), 1)
        prop = float(np.clip(leitura.nivel_cm / 100.0, 0, 1))
        ytopo = int(by + bh * (1 - prop))
        cv2.rectangle(img, (bx + 1, ytopo), (bx + bw - 1, by + bh - 1),
                      (200, 140, 40), -1)
        cv2.line(img, (bx - 5, ytopo), (bx + bw + 5, ytopo), cor, 2)
    return img


def processar_imagem(modelo, caminho, imgsz, salvar_em=None):
    res = modelo.predict(caminho, imgsz=imgsz, verbose=False)[0]
    numeros, gauges, surfaces = extrair(res, modelo.names)
    leitura = ler_nivel(numeros, gauges, surfaces)
    if salvar_em:
        cv2.imwrite(salvar_em, anotar(None, res, leitura, numeros))
    return leitura


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="hidrovision_v05.pt")
    ap.add_argument("--imagem")
    ap.add_argument("--pasta")
    ap.add_argument("--webcam", type=int)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--csv")
    ap.add_argument("--salvar-anotadas", help="pasta para salvar imagens anotadas")
    args = ap.parse_args()

    modelo = YOLO(args.modelo)
    print(f"modelo: {args.modelo} | classes: {len(modelo.names)}")

    # ---------- imagem única ----------
    if args.imagem:
        saida = None
        if args.salvar_anotadas:
            os.makedirs(args.salvar_anotadas, exist_ok=True)
            saida = os.path.join(args.salvar_anotadas,
                                 os.path.basename(args.imagem))
        r = processar_imagem(modelo, args.imagem, args.imgsz, saida)
        if r.nivel_cm is None:
            print(f"{args.imagem}: SEM LEITURA ({r.metodo}) — {r.detalhe}")
        else:
            ini, fim = faixa_do_nivel(r)
            print(f"{args.imagem}: faixa {ini}-{fim} cm "
                  f"(estimativa ~{r.nivel_cm:.0f} cm) | método {r.metodo} | "
                  f"conf {r.confianca:.2f} | {r.n_numeros} números | {r.detalhe}")
        return

    # ---------- pasta de imagens ----------
    if args.pasta:
        arquivos = sorted(sum([glob.glob(os.path.join(args.pasta, e))
                               for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG")], []))
        if not arquivos:
            print("nenhuma imagem encontrada em", args.pasta)
            return
        if args.salvar_anotadas:
            os.makedirs(args.salvar_anotadas, exist_ok=True)
        linhas, contagem = [], {}
        for f in arquivos:
            saida = (os.path.join(args.salvar_anotadas, os.path.basename(f))
                     if args.salvar_anotadas else None)
            r = processar_imagem(modelo, f, args.imgsz, saida)
            contagem[r.metodo] = contagem.get(r.metodo, 0) + 1
            linhas.append({"arquivo": os.path.basename(f),
                           "faixa_cm": ("" if r.nivel_cm is None
                                        else "%d-%d" % faixa_do_nivel(r)),
                           "nivel_cm": "" if r.nivel_cm is None else round(r.nivel_cm, 1),
                           "metodo": r.metodo, "confianca": round(r.confianca, 3),
                           "n_numeros": r.n_numeros,
                           "n_correcoes": len(r.correcoes),
                           "nao_detectados": " ".join(map(str, r.lacunas)),
                           "detalhe": r.detalhe})
            if r.nivel_cm is None:
                marca = "     sem leitura"
            else:
                ini, fim = faixa_do_nivel(r)
                marca = f"faixa {ini:3d}-{fim:3d} cm (~{r.nivel_cm:.0f})"
            print(f"{os.path.basename(f):40s} {marca}  [{r.metodo}]")
        print("\nresumo por método:")
        for k, v in sorted(contagem.items(), key=lambda t: -t[1]):
            print(f"  {k:15s} {v:4d}  ({v/len(arquivos)*100:.1f}%)")
        lidas = sum(1 for l in linhas if l["nivel_cm"] != "")
        print(f"leitura obtida em {lidas}/{len(arquivos)} imagens "
              f"({lidas/len(arquivos)*100:.1f}%)")
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as fp:
                w = csv.DictWriter(fp, fieldnames=list(linhas[0].keys()))
                w.writeheader()
                w.writerows(linhas)
            print("csv salvo em", args.csv)
        return

    # ---------- webcam ----------
    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.webcam, cv2.CAP_MSMF)
        if not cap.isOpened():
            print("não foi possível abrir a câmera", args.webcam)
            return
        filtro = FiltroMediana()
        print("q para sair")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = modelo.predict(frame, imgsz=args.imgsz, verbose=False)[0]
            numeros, gauges, surfaces = extrair(res, modelo.names)
            leitura = ler_nivel(numeros, gauges, surfaces)
            suave = filtro.add(leitura.nivel_cm)
            img = anotar(frame, res, leitura, numeros)
            if suave is not None:
                cv2.putText(img, f"mediana de 5 leituras: ~{suave:.0f} cm",
                            (12, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (245, 245, 245), 1)
            cv2.imshow("HidroVision", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    ap.error("informe --imagem, --pasta ou --webcam")


if __name__ == "__main__":
    main()
