# -*- coding: utf-8 -*-
"""
tendencia.py — HidroVision AI (Fase 3)
Tendência do nível em cm/h.

O problema que este módulo resolve: a leitura da régua pode vir em duas
resoluções distintas.

  - FINA (~2-3 cm): quando há dois ou mais números detectados, a escala px/cm
    sai do espaçamento de 10 cm entre eles e a linha d'água é interpolada.
  - GROSSA (degraus de 10 cm): quando só um número está visível, a leitura é
    o próprio valor do menor número — muda apenas quando a água cruza um
    múltiplo de 10.

Regressão linear simples funciona bem na resolução fina e falha na grossa: a
série vira uma escada e a inclinação oscila entre 0 e valores absurdos,
dependendo de quantos degraus caem dentro da janela.

A solução usada aqui:
  1. detecta automaticamente se a série está quantizada;
  2. na série fina, regressão sobre os valores, janela curta (30 min);
  3. na série grossa, mede o TEMPO ENTRE CRUZAMENTOS de degrau — se a água
     levou 40 min para ir de 50 para 60, a taxa é 10 cm / 0,667 h = 15 cm/h —
     com janela longa (2 h) e sem depender do valor bruto.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

BANDA_MORTA = 0.5       # cm/h — abaixo disso (em módulo) é "estável"
JANELA_FINA = 30        # minutos, para leitura interpolada
JANELA_GROSSA = 120     # minutos, para leitura em degraus de 10 cm
MIN_PONTOS = 4
PASSO_REGUA = 10.0      # cm entre marcações


@dataclass
class Tendencia:
    taxa_cm_h: float | None      # None = dados insuficientes
    rotulo: str                  # subindo | descendo | estavel | indefinido
    n_pontos: int
    r2: float = 0.0
    resolucao: str = "fina"      # 'fina' | 'quantizada'
    detalhe: str = ""

    @property
    def seta(self):
        return {"subindo": "\u2191", "descendo": "\u2193",
                "estavel": "\u2192", "indefinido": "?"}[self.rotulo]

    def __str__(self):
        if self.taxa_cm_h is None:
            return f"{self.rotulo} {self.seta}"
        return f"{self.taxa_cm_h:+.1f} cm/h {self.seta} ({self.rotulo})"


def _rotular(taxa, banda=BANDA_MORTA):
    if abs(taxa) < banda:
        return "estavel"
    return "subindo" if taxa > 0 else "descendo"


def detectar_quantizacao(valores, passo=PASSO_REGUA, tol=0.6):
    """
    True se os valores parecem múltiplos do passo da régua (leitura grossa).
    Usa a fração de leituras que caem sobre um múltiplo exato.
    """
    v = np.asarray(valores, dtype=float)
    if len(v) < 3:
        return False
    resto = np.abs(v - np.round(v / passo) * passo)
    return float(np.mean(resto < 0.05)) >= tol


def _taxa_por_cruzamentos(ts, valores, passo=PASSO_REGUA):
    """
    Taxa a partir do tempo entre cruzamentos de degrau.
    Retorna (taxa_cm_h, n_cruzamentos) ou (None, 0).
    """
    v = np.asarray(valores, dtype=float)
    t = pd.to_datetime(pd.Series(list(ts))).reset_index(drop=True)
    muda = np.flatnonzero(np.diff(v) != 0)
    if len(muda) == 0:
        return None, 0

    if len(muda) >= 2:
        i0, i1 = muda[0] + 1, muda[-1] + 1
        dt_h = (t.iloc[i1] - t.iloc[i0]).total_seconds() / 3600.0
        dv = v[i1] - v[i0]
        if dt_h > 0:
            return float(dv / dt_h), len(muda)
    # um único cruzamento: usa a janela inteira como base de tempo
    dt_h = (t.iloc[-1] - t.iloc[0]).total_seconds() / 3600.0
    dv = v[-1] - v[0]
    if dt_h <= 0:
        return None, len(muda)
    return float(dv / dt_h), len(muda)


def calcular(df, col_ts="ts", col_nivel="nivel_cm",
             janela_min=None, banda=BANDA_MORTA, passo=PASSO_REGUA):
    """
    df: DataFrame com timestamps (tz-aware) e nível — saída de
    Banco.serie_recente(). A janela é escolhida automaticamente conforme a
    resolução detectada, a menos que `janela_min` seja informado.
    """
    if df is None or df.empty:
        return Tendencia(None, "indefinido", 0, detalhe="sem leituras")

    ts_todos = pd.to_datetime(df[col_ts])
    quantizada = detectar_quantizacao(df[col_nivel].to_numpy(), passo)
    if janela_min is None:
        janela_min = JANELA_GROSSA if quantizada else JANELA_FINA

    fim = ts_todos.max()
    m = ts_todos >= fim - pd.Timedelta(minutes=janela_min)
    sub = df.loc[m]
    res = "quantizada" if quantizada else "fina"
    if len(sub) < MIN_PONTOS:
        return Tendencia(None, "indefinido", len(sub), resolucao=res,
                         detalhe=f"apenas {len(sub)} leituras na janela")

    t_sub = pd.to_datetime(sub[col_ts])
    y = sub[col_nivel].to_numpy(dtype=float)

    # ---------- leitura em degraus: taxa por cruzamentos ----------
    if quantizada:
        taxa, n_cruz = _taxa_por_cruzamentos(t_sub, y, passo)
        if taxa is None:
            # nenhum cruzamento: a água não andou um degrau inteiro na janela.
            # Isso é informação: |taxa| < passo / janela.
            limite = passo / (janela_min / 60.0)
            return Tendencia(0.0, "estavel", len(sub), resolucao=res,
                             detalhe=f"sem cruzar degrau em {janela_min} min "
                                     f"(|taxa| < {limite:.0f} cm/h)")
        return Tendencia(taxa, _rotular(taxa, banda), len(sub), resolucao=res,
                         detalhe=f"{n_cruz} cruzamento(s) em {janela_min} min")

    # ---------- leitura fina: regressão linear ----------
    t = (t_sub - t_sub.min()).dt.total_seconds().to_numpy() / 3600.0
    if np.allclose(y, y[0]):
        return Tendencia(0.0, "estavel", len(sub), 1.0, res,
                         "nível constante na janela")
    a, b = np.polyfit(t, y, 1)
    pred = a * t + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return Tendencia(float(a), _rotular(a, banda), len(sub), float(r2), res,
                     f"regressão em {janela_min} min")
