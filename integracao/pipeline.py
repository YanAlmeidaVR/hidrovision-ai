# -*- coding: utf-8 -*-
"""
pipeline.py — HidroVision AI (Fase 3)
Orquestrador: recebe cada leitura da câmera, grava no banco, calcula a
tendência, roda a previsão e avalia os alertas. É a função que o loop do
Raspberry Pi (e o dashboard) chamam.

Uso típico no loop da câmera:

    from fase3 import pipeline
    p = pipeline.Pipeline(modo="maquete", pasta_modelos="modelosPreditivos")
    ...
    resultado = p.processar_leitura(nivel_cm=48.2, metodo="geometria",
                                    confianca=0.71, menor_numero=50)
    # resultado traz nivel gravado, tendência, previsões e alertas gerados
"""
from dataclasses import dataclass, field

import pandas as pd

import banco as B
import tendencia as T
import alertas as A
import preditor as P


@dataclass
class Resultado:
    nivel_gravado: float | None
    tendencia: T.Tendencia | None
    previsoes: dict | None
    eventos_alerta: list = field(default_factory=list)
    projecao: object = None


class Pipeline:
    def __init__(self, modo="maquete", db=B.DB_PADRAO,
                 pasta_modelos=".", canais=None,
                 prever_a_cada_min=10, janela_mediana=3):
        self.banco = B.Banco(db)
        self.alertas = A.GerenciadorAlertas(
            modo=modo, banco=self.banco,
            canais=canais or [A.CanalConsole(), A.CanalArquivo()])
        try:
            self.preditor = P.Preditor(pasta_modelos)
        except FileNotFoundError as e:
            print(f"[aviso] {e} — rodando sem previsão")
            self.preditor = None
        self.prever_a_cada = pd.Timedelta(minutes=prever_a_cada_min)
        # janela do filtro de mediana. 3 é o equilíbrio para leitura em degraus:
        # ainda descarta uma detecção espúria isolada, mas confirma um degrau
        # real em 2 leituras em vez de 3 (menos atraso no alerta).
        self.janela_mediana = janela_mediana
        self._ultima_projecao = None
        self._ultima_previsao_ts = None
        self._ultimas_previsoes = None

    # ------------------------------------------------------------------
    def processar_leitura(self, nivel_cm, metodo="", confianca=None,
                          menor_numero=None, ts=None):
        """Chamada a cada leitura da câmera. Faz todo o ciclo."""
        gravado = self.banco.gravar_leitura(
            nivel_cm, metodo, confianca, menor_numero, ts,
            janela_mediana=self.janela_mediana)
        if gravado is None:
            return Resultado(None, None, None)

        recente = self.banco.serie_recente(horas=3)
        tend = T.calcular(recente)

        previsoes = self._prever_se_necessario(ts)

        eventos, proj = self.alertas.avaliar(
            gravado, tendencia=tend, df_recente=recente, ts=ts)
        self._ultima_projecao = proj

        return Resultado(gravado, tend, previsoes, eventos, proj)

    def _prever_se_necessario(self, ts):
        """Previsão é cara relativa à leitura: roda a cada N minutos, não a
        cada frame."""
        if self.preditor is None:
            return None
        agora = pd.Timestamp(ts) if ts is not None else pd.Timestamp.now(tz=B.FUSO)
        if (self._ultima_previsao_ts is not None
                and agora - self._ultima_previsao_ts < self.prever_a_cada):
            return self._ultimas_previsoes
        serie = self.banco.serie_horaria(horas=30 * 24, ate=agora)
        self._ultimas_previsoes = self.preditor.prever(serie)
        self._ultima_previsao_ts = agora
        return self._ultimas_previsoes

    # ------------------------------------------------------------------
    # utilidades para o dashboard
    # ------------------------------------------------------------------
    def estado_atual(self):
        """Snapshot para o card de status do dashboard."""
        ult = self.banco.ultima_leitura()
        recente = self.banco.serie_recente(horas=3)
        tend = T.calcular(recente)
        proj = self._ultima_projecao
        if proj is None and ult is not None:
            import projecao as PJ
            proj = PJ.projetar(ult["nivel_cm"], tend, recente,
                               nivel_critico=self.alertas.nivel_critico)
        return {"ultima_leitura": ult, "tendencia": tend, "projecao": proj,
                "previsoes": self._ultimas_previsoes,
                "modo": self.alertas.modo,
                "limiares": self.alertas.limiares,
                "nivel_critico": self.alertas.nivel_critico}

    def simular_chuva(self, chuva_mmh, horas_de_chuva=6):
        if self.preditor is None:
            return None
        serie = self.banco.serie_horaria(horas=30 * 24)
        return self.preditor.simular_chuva(serie, chuva_mmh, horas_de_chuva)

    def trocar_modo(self, modo):
        self.alertas.trocar_modo(modo)
