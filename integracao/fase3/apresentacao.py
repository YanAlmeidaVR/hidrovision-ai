# -*- coding: utf-8 -*-
"""
apresentacao.py — HidroVision AI

Modo de apresentação da FETIN. Mostra as duas camadas lado a lado:

  RIO (dados reais da estação da ANA)  ->  antecipa a cheia
  RÉGUA URBANA (maquete com câmera)    ->  confirma a chegada da água

A previsão de chuva vem de um arquivo congelado (cenario_chuva.json), não da
internet: numa feira, depender de rede é risco desnecessário. O arquivo é
preparado antes e descreve o cenário narrado na apresentação.

A cheia do rio é um evento REAL reproduzido a partir do histórico. Não é
simulação: aconteceu, está nos dados da ANA, e o sistema reage a ela.

Uso:
    # ensaio completo, sem câmera (a régua também é simulada)
    python apresentacao.py --dados dados_treino.csv --modelos ../preditivo/modelos

    # no dia: a régua vem da câmera
    python apresentacao.py --dados dados_treino.csv --modelos ../preditivo/modelos --camera

    # ajustar a velocidade (padrão: 1 hora de dados por segundo)
    python apresentacao.py --dados dados_treino.csv --seg-por-hora 0.5
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import banco as B
import tendencia as T
import projecao as PJ
import preditor as P
import clima as C
import monitor as M

# trecho do histórico usado: a cheia real de março de 2026
INICIO_EVENTO = "2026-03-10 12:00"
FIM_EVENTO = "2026-03-12 12:00"
HORAS_HISTORICO = 72          # contexto anterior, para os lags do modelo

LARG = 78


def barra(valor, maximo, largura=22, cheio="█", vazio="·"):
    n = int(round(largura * min(max(valor / maximo, 0), 1)))
    return cheio * n + vazio * (largura - n)


def cabecalho(texto):
    print(f"\n{'=' * LARG}")
    print(f" {texto}")
    print("=" * LARG)


class Apresentacao:
    def __init__(self, csv, pasta_modelos, cenario, db="apresentacao.db",
                 com_camera=False):
        if os.path.exists(db):
            os.remove(db)
        self.banco = B.Banco(db)
        self.preditor = P.Preditor(pasta_modelos)
        self.clima = C.ClimaFixo(cenario)
        self.com_camera = com_camera

        # série completa do histórico
        df = pd.read_csv(csv, parse_dates=["datahora"])
        df["datahora"] = pd.to_datetime(df["datahora"], utc=True).dt.tz_convert(B.FUSO)
        self.serie = df.dropna(subset=["nivel_cm"]).set_index("datahora")

        ini = pd.Timestamp(INICIO_EVENTO, tz=B.FUSO)
        fim = pd.Timestamp(FIM_EVENTO, tz=B.FUSO)
        self.evento = self.serie.loc[ini:fim]
        contexto = self.serie.loc[ini - pd.Timedelta(hours=HORAS_HISTORICO):ini]

        # semeia o banco com o contexto anterior ao evento
        for ts, row in contexto.iterrows():
            self.banco.gravar_leitura(float(row["nivel_cm"]), "estacao_ana",
                                      ts=ts, fonte="ana", janela_mediana=1)

        self.nivel_regua = None
        self.leitor = None
        if com_camera:
            self._abrir_camera()

    # ------------------------------------------------------------------
    def _abrir_camera(self):
        """Conecta a leitura da régua ao módulo de visão, se disponível."""
        try:
            import cv2
            from ultralytics import YOLO
            import sys
            sys.path.insert(0, os.path.join("..", "visao"))
            import mdYOLO as V
            self.cv2, self.V = cv2, V
            self.modelo = YOLO(os.path.join("..", "visao", "hidrovision_v05.pt"))
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.leitor = True
            print("câmera conectada")
        except Exception as e:
            print(f"[aviso] câmera indisponível ({e}); a régua será simulada")
            self.leitor = None

    def ler_regua(self, fracao):
        """
        Nível da régua urbana. Com câmera, lê do vídeo; sem, simula a maquete
        enchendo proporcionalmente ao avanço do evento.
        """
        if self.leitor:
            ok, frame = self.cap.read()
            if ok:
                res = self.modelo.predict(frame, imgsz=640, verbose=False)[0]
                nums, gs, ss = self.V.extrair(res, self.modelo.names)
                leitura = self.V.ler_nivel(nums, gs, ss)
                if leitura.nivel_cm is not None:
                    return leitura.nivel_cm
            return self.nivel_regua
        # sem câmera: a maquete acompanha o rio, em degraus de 10 cm
        return float(int((10 + 85 * fracao) // 10) * 10)

    # ------------------------------------------------------------------
    def rodar(self, seg_por_hora=1.0):
        resumo = self.clima.resumo()
        cabecalho("HIDROVISION AI — MONITORAMENTO EM TEMPO REAL")
        print(f" Estação 61305000 · Rio Sapucaí · Santa Rita do Sapucaí — MG")
        print(f" Previsão: {C.descrever(resumo)}")
        print(f"\n Reproduzindo um evento real registrado pela ANA em março de 2026.")
        print(f" A régua urbana é lida "
              f"{'pela câmera' if self.leitor else 'da maquete simulada'}.")

        total = len(self.evento)
        alertado = set()

        for i, (ts, row) in enumerate(self.evento.iterrows()):
            fracao = i / max(total - 1, 1)
            nivel_rio = float(row["nivel_cm"])
            self.banco.gravar_leitura(nivel_rio, "estacao_ana", ts=ts,
                                      fonte="ana", janela_mediana=1)

            # ---- camada 1: previsão do rio ----
            serie = self.banco.serie_horaria(horas=30 * 24, ate=ts)
            prev_sem = self.preditor.prever(serie)
            prev_com = None
            if prev_sem and resumo.get("media_mmh", 0) > 0.05:
                prev_com = self.preditor.simular_chuva(
                    serie, resumo["media_mmh"], horas_de_chuva=24)

            # ---- camada 2: régua urbana ----
            self.nivel_regua = self.ler_regua(fracao)
            recente = self.banco.serie_recente(horas=3)
            tend_regua = T.calcular(recente)
            proj = PJ.projetar(self.nivel_regua, tend_regua)

            risco = M.avaliar_risco(prev_sem, prev_com, resumo,
                                    self.nivel_regua, proj)

            self._exibir(ts, nivel_rio, prev_sem, prev_com, proj, risco)

            chave = (risco.nivel, len(risco.motivos))
            if risco.nivel in ("alerta", "emergencia") and chave not in alertado:
                alertado.add(chave)
            time.sleep(seg_por_hora)

        cabecalho("FIM DA REPRODUÇÃO")

    def _exibir(self, ts, nivel_rio, prev_sem, prev_com, proj, risco):
        cores = {"normal": "\033[92m", "atencao": "\033[93m",
                 "alerta": "\033[91m", "emergencia": "\033[1;91m"}
        reset = "\033[0m"
        c = cores.get(risco.nivel, "")

        print(f"\n{'-' * LARG}")
        print(f" {ts:%d/%m %H:%M}")

        # rio
        print(f"   RIO    {nivel_rio:6.0f} cm  {barra(nivel_rio, 450)}")
        if prev_sem:
            linha = f"          previsão 24 h: {prev_sem.get('24h', 0):.0f} cm"
            if prev_com:
                linha += f"   com a chuva prevista: {prev_com.get('24h', 0):.0f} cm"
            print(linha)

        # régua urbana
        if self.nivel_regua is not None:
            folga = 100 - self.nivel_regua
            print(f"   RÉGUA  {self.nivel_regua:6.0f} cm  "
                  f"{barra(self.nivel_regua, 100)}  folga {folga:.0f} cm")
            if proj.estado == "subindo":
                print(f"          {proj.taxa_cm_h:+.0f} cm/h — atinge a área "
                      f"urbana em {proj.tempo_formatado}")

        print(f"   {c}[{risco.nivel.upper()}] {risco.titulo}{reset}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="dados_treino.csv")
    ap.add_argument("--modelos", default="../preditivo/modelos")
    ap.add_argument("--cenario", default="cenario_chuva.json")
    ap.add_argument("--camera", action="store_true",
                    help="lê a régua da câmera em vez de simular")
    ap.add_argument("--seg-por-hora", type=float, default=1.0,
                    help="segundos de exibição por hora de dados")
    args = ap.parse_args()

    a = Apresentacao(args.dados, args.modelos, args.cenario,
                     com_camera=args.camera)
    a.rodar(args.seg_por_hora)
