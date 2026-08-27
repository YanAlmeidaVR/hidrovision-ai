# -*- coding: utf-8 -*-
"""
projecao.py — HidroVision AI (Fase 3)

A régua não mede o rio: ela fica no ponto onde a água chega à área urbana.
O zero é o nível de referência e os 100 cm são o ponto crítico — quando a
lâmina alcança essa marca, a água atinge a cidade.

Logo, a pergunta que o sistema precisa responder não é "qual será o nível
daqui a 6 h", e sim: QUANTO TEMPO FALTA PARA A ÁGUA ATINGIR OS 100 cm.

Isso é uma extrapolação da tendência, não um modelo estatístico — e é
justamente por isso que é defensável: a conta é explicável ("subindo 15 cm/h,
faltam 40 cm, logo ~2h40") e usa apenas a resolução que a régua oferece.

Três estados possíveis:
  SUBINDO  -> projeta o tempo restante até o nível crítico
  ESTÁVEL  -> não projeta (a extrapolação não se aplica)
  DESCENDO -> projeta quando a água sai da régua, mas sem urgência
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

NIVEL_CRITICO = 100.0      # cm — a água atinge a área urbana
TAXA_MINIMA = 0.5          # cm/h — abaixo disso não há trajetória definida
HORIZONTE_MAX = 24.0       # h — acima disso a projeção não é informativa

# faixas de urgência por tempo restante (horas)
FAIXAS_TEMPO = {"atencao": 6.0, "alerta": 3.0, "emergencia": 1.0}


@dataclass
class Projecao:
    estado: str                     # subindo | estavel | descendo | indefinido
    nivel_cm: float | None
    taxa_cm_h: float | None
    folga_cm: float | None          # quanto falta até o nível crítico
    horas_para_critico: float | None
    aceleracao_cm_h2: float | None = None   # >0 = subida acelerando
    urgencia: str | None = None     # emergencia | alerta | atencao | None
    detalhe: str = ""

    @property
    def tempo_formatado(self):
        h = self.horas_para_critico
        if h is None:
            return "—"
        if h >= HORIZONTE_MAX:
            return f"mais de {int(HORIZONTE_MAX)} h"
        # arredonda primeiro em minutos e só então separa em horas:
        # fazer o contrário produzia saídas como "10h60" em vez de "11 h"
        total_min = int(round(h * 60))
        if total_min < 60:
            return f"~{total_min} min"
        horas, minutos = divmod(total_min, 60)
        return f"~{horas}h{minutos:02d}" if minutos else f"~{horas} h"

    def resumo(self):
        if self.nivel_cm is None:
            return "sem leitura suficiente"
        base = f"água em {self.nivel_cm:.0f} cm"
        if self.estado == "subindo":
            txt = (f"{base}, subindo {self.taxa_cm_h:.0f} cm/h -> atinge a área "
                   f"urbana em {self.tempo_formatado} "
                   f"(folga de {self.folga_cm:.0f} cm)")
            if self.aceleracao_cm_h2 and self.aceleracao_cm_h2 > 2:
                txt += " | subida ACELERANDO"
            return txt
        if self.estado == "descendo":
            return (f"{base}, recuando {abs(self.taxa_cm_h):.0f} cm/h "
                    f"(folga de {self.folga_cm:.0f} cm)")
        if self.estado == "estavel":
            return f"{base}, estável (folga de {self.folga_cm:.0f} cm)"
        return f"{base}, tendência indefinida"


def _urgencia_por_tempo(horas):
    if horas is None:
        return None
    for nome in ("emergencia", "alerta", "atencao"):
        if horas <= FAIXAS_TEMPO[nome]:
            return nome
    return None


def calcular_aceleracao(df, col_ts="ts", col_nivel="nivel_cm",
                        janela_min=60):
    """
    Compara a taxa da metade recente com a da metade anterior da janela.
    Positivo = a subida está acelerando (extrapolação linear subestima).
    """
    if df is None or len(df) < 8:
        return None
    ts = pd.to_datetime(df[col_ts])
    fim = ts.max()
    ini = fim - pd.Timedelta(minutes=janela_min)
    meio = ini + (fim - ini) / 2
    a1 = df.loc[(ts >= ini) & (ts < meio)]
    a2 = df.loc[ts >= meio]
    if len(a1) < 3 or len(a2) < 3:
        return None

    def taxa(sub):
        t = (pd.to_datetime(sub[col_ts]) - ini).dt.total_seconds().to_numpy() / 3600
        y = sub[col_nivel].to_numpy(dtype=float)
        if np.allclose(y, y[0]) or len(np.unique(t)) < 2:
            return 0.0
        return float(np.polyfit(t, y, 1)[0])

    dt_h = (fim - meio).total_seconds() / 3600
    if dt_h <= 0:
        return None
    return (taxa(a2) - taxa(a1)) / dt_h


def projetar(nivel_cm, tendencia, df_recente=None,
             nivel_critico=NIVEL_CRITICO):
    """
    nivel_cm   : nível atual lido na régua
    tendencia  : objeto Tendencia (de tendencia.calcular)
    df_recente : série recente, para estimar aceleração (opcional)

    A aceleração só é estimada quando a leitura é fina. Em leitura
    quantizada (degraus de 10 cm) o cálculo compara duas regressões sobre uma
    escada e produz valores espúrios — reportaria "acelerando" numa subida
    perfeitamente linear.
    """
    if nivel_cm is None or tendencia is None:
        return Projecao("indefinido", nivel_cm, None, None, None,
                        detalhe="sem leitura ou sem tendência")

    folga = float(nivel_critico - nivel_cm)
    taxa = tendencia.taxa_cm_h
    leitura_fina = getattr(tendencia, "resolucao", "fina") == "fina"
    acel = (calcular_aceleracao(df_recente)
            if (df_recente is not None and leitura_fina) else None)

    if taxa is None:
        return Projecao("indefinido", nivel_cm, None, folga, None, acel,
                        detalhe=tendencia.detalhe or "tendência indefinida")

    # --- estável: a extrapolação não se aplica ---
    if abs(taxa) < TAXA_MINIMA:
        return Projecao("estavel", nivel_cm, taxa, folga, None, acel,
                        detalhe="nível estável — projeção não se aplica")

    # --- descendo: informa o recuo, sem urgência de trajetória ---
    if taxa < 0:
        horas_saida = abs(nivel_cm / taxa) if nivel_cm > 0 else 0.0
        return Projecao("descendo", nivel_cm, taxa, folga, None, acel,
                        detalhe=f"recuo; sai da régua em ~{horas_saida:.1f} h "
                                f"no ritmo atual")

    # --- subindo: projeta o tempo até o nível crítico ---
    if folga <= 0:
        return Projecao("subindo", nivel_cm, taxa, folga, 0.0, acel,
                        urgencia="emergencia",
                        detalhe="nível crítico já atingido")
    horas = folga / taxa
    urg = _urgencia_por_tempo(horas)
    det = f"extrapolação linear a {taxa:.1f} cm/h"
    if acel is not None and acel > 2:
        det += f"; aceleração de {acel:+.1f} cm/h² (tempo real pode ser menor)"
    elif acel is not None and acel < -2:
        det += f"; desaceleração de {acel:+.1f} cm/h² (tempo real pode ser maior)"
    return Projecao("subindo", nivel_cm, taxa, folga, horas, acel, urg, det)
