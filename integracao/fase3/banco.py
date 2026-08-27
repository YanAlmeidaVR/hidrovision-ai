# -*- coding: utf-8 -*-
"""
banco.py — HidroVision AI (Fase 3)
Camada de dados em SQLite: leituras da câmera, série horária e alertas.

Duas tabelas:
  leituras : cada leitura da câmera (a cada 30-60 s)
  alertas  : histórico de disparos/rearmes

Regras embutidas:
  - filtro de mediana ANTES de gravar (outlier não entra no histórico);
  - agregação horária para alimentar o preditor (os modelos XGBoost foram
    treinados com dado horário — alimentar com dado de 30 s quebra os lags);
  - importação da série histórica da ANA para o mesmo banco (fonte única).
"""
import os
import sqlite3
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

FUSO = "America/Sao_Paulo"
DB_PADRAO = "hidrovision.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS leituras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,               -- ISO 8601 com fuso
    nivel_cm REAL NOT NULL,
    metodo TEXT,
    confianca REAL,
    menor_numero INTEGER,
    fonte TEXT DEFAULT 'camera'     -- 'camera' | 'ana'
);
CREATE INDEX IF NOT EXISTS idx_leituras_ts ON leituras(ts);

CREATE TABLE IF NOT EXISTS alertas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    tipo TEXT NOT NULL,             -- 'atencao' | 'alerta' | 'emergencia'
    estado TEXT NOT NULL,           -- 'disparado' | 'rearmado'
    nivel_cm REAL,
    origem TEXT,                    -- 'nivel_atual' | 'previsao_6h' | ...
    mensagem TEXT
);
CREATE INDEX IF NOT EXISTS idx_alertas_ts ON alertas(ts);

CREATE TABLE IF NOT EXISTS previsoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,               -- instante em que a previsão foi feita
    nivel_atual REAL,
    prev_6h REAL, prev_12h REAL, prev_24h REAL,             -- sem chuva nova
    prev_6h_chuva REAL, prev_12h_chuva REAL, prev_24h_chuva REAL,
    chuva_total_mm REAL,            -- previsão meteorológica do momento
    chuva_media_mmh REAL,
    chuva_pico_mmh REAL,
    chuva_prob_max INTEGER,
    risco TEXT
);
CREATE INDEX IF NOT EXISTS idx_previsoes_ts ON previsoes(ts);
"""


def _agora():
    """Timestamp atual no fuso do projeto."""
    return pd.Timestamp.now(tz=FUSO)


class Banco:
    def __init__(self, caminho=DB_PADRAO):
        self.caminho = caminho
        self.con = sqlite3.connect(caminho)
        self.con.executescript(_SCHEMA)
        self.con.commit()
        self._buffer_mediana = []

    def fechar(self):
        self.con.close()

    # ------------------------------------------------------------------
    # gravação de leituras (com filtro de mediana embutido)
    # ------------------------------------------------------------------
    def gravar_leitura(self, nivel_cm, metodo="", confianca=None,
                       menor_numero=None, ts=None, janela_mediana=3,
                       fonte="camera"):
        """
        Grava uma leitura. O valor gravado é a MEDIANA das últimas
        `janela_mediana` leituras (incluindo esta), de modo que um outlier
        isolado não entra no histórico. Retorna o valor efetivamente gravado.
        """
        if nivel_cm is None:
            return None
        self._buffer_mediana.append(float(nivel_cm))
        if len(self._buffer_mediana) > janela_mediana:
            self._buffer_mediana.pop(0)
        valor = float(np.median(self._buffer_mediana))

        ts = ts if ts is not None else _agora()
        self.con.execute(
            "INSERT INTO leituras (ts, nivel_cm, metodo, confianca, "
            "menor_numero, fonte) VALUES (?,?,?,?,?,?)",
            (pd.Timestamp(ts).isoformat(), valor, metodo, confianca,
             menor_numero, fonte))
        self.con.commit()
        return valor

    # ------------------------------------------------------------------
    # consultas
    # ------------------------------------------------------------------
    def serie_recente(self, horas=48, fonte=None):
        """
        Leituras cruas das últimas N horas -> DataFrame [ts, nivel_cm, ...].
        A janela é ancorada na ÚLTIMA LEITURA, não no relógio de parede —
        assim funciona igual em operação, replay e simulação.
        """
        ult = self.con.execute("SELECT MAX(ts) FROM leituras").fetchone()[0]
        if ult is None:
            return pd.DataFrame(columns=["ts", "nivel_cm", "metodo",
                                         "confianca", "fonte"])
        corte = (pd.Timestamp(ult) - timedelta(hours=horas)).isoformat()
        q = "SELECT ts, nivel_cm, metodo, confianca, fonte FROM leituras WHERE ts >= ?"
        args = [corte]
        if fonte:
            q += " AND fonte = ?"
            args.append(fonte)
        df = pd.read_sql_query(q + " ORDER BY ts", self.con, params=args)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(FUSO)
        return df

    def serie_horaria(self, horas=27 * 24, ate=None):
        """
        Série agregada em base HORÁRIA (média das leituras de cada hora),
        no formato que o features.py espera: index datahora, coluna nivel_cm.
        É esta função que alimenta o preditor.
        """
        fim = pd.Timestamp(ate).tz_convert(FUSO) if ate is not None else _agora()
        ini = fim - timedelta(hours=horas)
        df = pd.read_sql_query(
            "SELECT ts, nivel_cm FROM leituras WHERE ts >= ? AND ts <= ? ORDER BY ts",
            self.con, params=[ini.isoformat(), fim.isoformat()])
        if df.empty:
            return pd.DataFrame(columns=["nivel_cm"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(FUSO)
        serie = (df.set_index("ts")["nivel_cm"]
                   .resample("1h").mean())
        # grade contínua (buracos ficam NaN — o features.py sabe lidar)
        grade = pd.date_range(serie.index.min(), serie.index.max(),
                              freq="1h", tz=FUSO)
        out = serie.reindex(grade).to_frame("nivel_cm")
        out.index.name = "datahora"
        return out

    def ultima_leitura(self):
        row = self.con.execute(
            "SELECT ts, nivel_cm, metodo, confianca FROM leituras "
            "ORDER BY ts DESC LIMIT 1").fetchone()
        if not row:
            return None
        return {"ts": pd.Timestamp(row[0]).tz_convert(FUSO),
                "nivel_cm": row[1], "metodo": row[2], "confianca": row[3]}

    # ------------------------------------------------------------------
    # alertas
    # ------------------------------------------------------------------
    def gravar_alerta(self, tipo, estado, nivel_cm, origem, mensagem="", ts=None):
        ts = ts if ts is not None else _agora()
        self.con.execute(
            "INSERT INTO alertas (ts, tipo, estado, nivel_cm, origem, mensagem) "
            "VALUES (?,?,?,?,?,?)",
            (pd.Timestamp(ts).isoformat(), tipo, estado, nivel_cm, origem, mensagem))
        self.con.commit()

    def gravar_previsao(self, prev_sem, prev_com=None, clima=None,
                        risco=None, ts=None):
        """Registra um ciclo de previsão, para o dashboard montar o histórico."""
        if not prev_sem:
            return
        ts = ts if ts is not None else _agora()
        c = clima or {}
        self.con.execute(
            "INSERT INTO previsoes (ts, nivel_atual, prev_6h, prev_12h, prev_24h,"
            " prev_6h_chuva, prev_12h_chuva, prev_24h_chuva, chuva_total_mm,"
            " chuva_media_mmh, chuva_pico_mmh, chuva_prob_max, risco)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pd.Timestamp(ts).isoformat(),
             prev_sem.get("nivel_atual"),
             prev_sem.get("6h"), prev_sem.get("12h"), prev_sem.get("24h"),
             (prev_com or {}).get("6h"), (prev_com or {}).get("12h"),
             (prev_com or {}).get("24h"),
             c.get("total_mm"), c.get("media_mmh"), c.get("pico_mmh"),
             c.get("prob_max"), risco))
        self.con.commit()

    def previsoes_recentes(self, horas=72):
        """Histórico de previsões, para o gráfico do dashboard."""
        ult = self.con.execute("SELECT MAX(ts) FROM previsoes").fetchone()[0]
        if ult is None:
            return pd.DataFrame()
        corte = (pd.Timestamp(ult) - timedelta(hours=horas)).isoformat()
        df = pd.read_sql_query(
            "SELECT * FROM previsoes WHERE ts >= ? ORDER BY ts",
            self.con, params=[corte])
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(FUSO)
        return df

    def ultima_previsao(self):
        df = pd.read_sql_query(
            "SELECT * FROM previsoes ORDER BY ts DESC LIMIT 1", self.con)
        if df.empty:
            return None
        d = df.iloc[0].to_dict()
        d["ts"] = pd.Timestamp(d["ts"]).tz_convert(FUSO)
        return d

    def alertas_recentes(self, n=20):
        df = pd.read_sql_query(
            "SELECT ts, tipo, estado, nivel_cm, origem, mensagem FROM alertas "
            "ORDER BY ts DESC LIMIT ?", self.con, params=[n])
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(FUSO)
        return df

    # ------------------------------------------------------------------
    # importação do histórico da ANA (fonte única de verdade)
    # ------------------------------------------------------------------
    def importar_csv_ana(self, caminho_csv, fonte="ana"):
        """
        Importa o dados_treino.csv (ou ana_horario.csv) para a tabela de
        leituras, uma linha por hora. Idempotente: apaga a fonte antes.
        """
        df = pd.read_csv(caminho_csv, parse_dates=["datahora"])
        df["datahora"] = pd.to_datetime(df["datahora"], utc=True).dt.tz_convert(FUSO)
        df = df.dropna(subset=["nivel_cm"])
        self.con.execute("DELETE FROM leituras WHERE fonte = ?", (fonte,))
        self.con.executemany(
            "INSERT INTO leituras (ts, nivel_cm, metodo, fonte) VALUES (?,?,?,?)",
            [(ts.isoformat(), float(n), "estacao_ana", fonte)
             for ts, n in zip(df["datahora"], df["nivel_cm"])])
        self.con.commit()
        return len(df)
