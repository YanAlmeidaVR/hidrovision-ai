# -*- coding: utf-8 -*-
"""
preditor.py — HidroVision AI (Fase 3)
Carrega os modelos XGBoost e prevê o nível futuro a partir da série horária
do banco. Também expõe a simulação de chuva (usada pelo slider do dashboard).

Modelos esperados (na pasta indicada):
  modelo_delta_6h.json / modelo_delta_12h.json / modelo_delta_24h.json
Eles preveem a VARIAÇÃO do nível; o nível previsto é nivel_atual + delta.
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb

HORIZONTES = (6, 12, 24)


def _construir_features(df):
    """
    Mesmas features do treino (features do notebook de horizontes longos).
    df: DataFrame indexado por datahora horária com colunas nivel_cm e
    (opcional) chuva_mm.
    """
    X = pd.DataFrame(index=df.index)
    nivel = df["nivel_cm"]
    chuva = df.get("chuva_mm", pd.Series(0.0, index=df.index)).fillna(0.0)

    X["nivel"] = nivel
    for h in (1, 2, 3, 6, 12, 24):
        X[f"nivel_lag{h}"] = nivel.shift(h)
    X["delta_1h"] = nivel - nivel.shift(1)
    X["delta_3h"] = nivel - nivel.shift(3)
    X["delta_6h"] = nivel - nivel.shift(6)
    X["tend_6h"] = X["delta_6h"] / 6.0

    X["chuva"] = chuva
    for j in (3, 6, 12, 24, 48, 72):
        X[f"chuva_acum{j}"] = chuva.rolling(j, min_periods=1).sum()
    X["chuva_max6"] = chuva.rolling(6, min_periods=1).max()
    houve = (chuva > 1.0).astype(int)
    g = houve.cumsum()
    X["horas_sem_chuva"] = (houve.groupby(g).cumcount()
                            .where(g > 0, 72).clip(upper=72))
    X["chuva24_x_nivel"] = X["chuva_acum24"] * X["nivel"] / 100.0

    mes = X.index.month
    X["mes_sen"] = np.sin(2 * np.pi * mes / 12)
    X["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    return X


class Preditor:
    def __init__(self, pasta_modelos="."):
        self.modelos = {}
        for h in HORIZONTES:
            caminho = os.path.join(pasta_modelos, f"modelo_delta_{h}h.json")
            if os.path.exists(caminho):
                m = xgb.XGBRegressor()
                m.load_model(caminho)
                self.modelos[h] = m
        if not self.modelos:
            raise FileNotFoundError(
                f"nenhum modelo_delta_*.json encontrado em {pasta_modelos}")
        # ordem exata das features que o modelo espera
        self.feats = self.modelos[list(self.modelos)[0]].get_booster().feature_names

    # ------------------------------------------------------------------
    def _linha_atual(self, serie_horaria):
        """Última linha de features completa da série horária do banco."""
        if serie_horaria is None or serie_horaria.empty:
            return None
        X = _construir_features(serie_horaria)
        X = X[self.feats].dropna()
        if X.empty:
            return None
        return X.iloc[-1]

    def prever(self, serie_horaria):
        """
        serie_horaria: saída de Banco.serie_horaria() (>= 25 h de dados).
        Retorna {'nivel_atual': x, '6h': y, '12h': z, '24h': w} em cm,
        ou None se não houver histórico suficiente.
        """
        linha = self._linha_atual(serie_horaria)
        if linha is None:
            return None
        out = {"nivel_atual": float(linha["nivel"])}
        for h, m in self.modelos.items():
            delta = float(m.predict(pd.DataFrame([linha])[self.feats])[0])
            out[f"{h}h"] = round(float(linha["nivel"]) + delta, 1)
        return out

    # ------------------------------------------------------------------
    def simular_chuva(self, serie_horaria, chuva_mmh, horas_de_chuva=6):
        """
        Slider do dashboard: injeta uma chuva hipotética nas features e
        devolve as previsões. chuva_mmh em mm/h.
        """
        linha = self._linha_atual(serie_horaria)
        if linha is None:
            return None
        L = linha.copy()
        L["chuva"] = chuva_mmh
        for j in (3, 6, 12, 24, 48, 72):
            L[f"chuva_acum{j}"] = L[f"chuva_acum{j}"] + min(j, horas_de_chuva) * chuva_mmh
        L["chuva_max6"] = max(L["chuva_max6"], chuva_mmh)
        if chuva_mmh > 1:
            L["horas_sem_chuva"] = 0
        L["chuva24_x_nivel"] = L["chuva_acum24"] * L["nivel"] / 100.0

        out = {"nivel_atual": float(L["nivel"]), "chuva_mmh": chuva_mmh}
        for h, m in self.modelos.items():
            delta = float(m.predict(pd.DataFrame([L])[self.feats])[0])
            out[f"{h}h"] = round(float(L["nivel"]) + delta, 1)
        return out
