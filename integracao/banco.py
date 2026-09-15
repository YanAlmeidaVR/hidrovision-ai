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
    ts TEXT NOT NULL,
    nivel_cm REAL NOT NULL,
    metodo TEXT,
    confianca REAL,
    menor_numero INTEGER,
    fonte TEXT DEFAULT 'camera',
    chuva_mm REAL
);
CREATE INDEX IF NOT EXISTS idx_leituras_ts ON leituras(ts);
CREATE UNIQUE INDEX IF NOT EXISTS ix_leituras_ts_fonte ON leituras(ts, fonte);
CREATE INDEX IF NOT EXISTS idx_leituras_epoch_fonte
    ON leituras(fonte, strftime('%s', ts));

CREATE TABLE IF NOT EXISTS alertas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    tipo TEXT NOT NULL,
    estado TEXT NOT NULL,
    nivel_cm REAL,
    origem TEXT,
    mensagem TEXT
);
CREATE INDEX IF NOT EXISTS idx_alertas_ts ON alertas(ts);

CREATE TABLE IF NOT EXISTS previsoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    nivel_atual REAL,
    prev_6h REAL, prev_12h REAL, prev_24h REAL,
    prev_6h_chuva REAL, prev_12h_chuva REAL, prev_24h_chuva REAL,
    chuva_total_mm REAL,
    chuva_media_mmh REAL,
    chuva_pico_mmh REAL,
    chuva_prob_max INTEGER,
    risco TEXT,
    ana_ultima_leitura TEXT,
    ana_horas_novas INTEGER,
    ana_fora_do_ar INTEGER,
    bacia_horas_previstas INTEGER,
    bacia_horas_chuva INTEGER,
    cidade_mmh REAL,
    cidade_acum6h REAL,
    estacao_mmh REAL,
    estacao_acum6h REAL,
    risco_motivos TEXT
);
CREATE INDEX IF NOT EXISTS idx_previsoes_ts ON previsoes(ts);
CREATE UNIQUE INDEX IF NOT EXISTS ix_previsoes_ts_unico ON previsoes(ts);
"""

_VIEW_ERRO = """
CREATE VIEW previsoes_erro AS
WITH alvo AS (
    SELECT ts AS ts_ciclo, 6 AS horizonte, datetime(ts, '+6 hours') AS ts_alvo,
           nivel_atual, prev_6h AS previsto_sem_chuva,
           prev_6h_chuva AS previsto_com_chuva
    FROM previsoes
    UNION ALL
    SELECT ts, 12, datetime(ts, '+12 hours'), nivel_atual, prev_12h, prev_12h_chuva
    FROM previsoes
    UNION ALL
    SELECT ts, 24, datetime(ts, '+24 hours'), nivel_atual, prev_24h, prev_24h_chuva
    FROM previsoes
),
cand AS (
    SELECT a.*,
        (SELECT ts FROM leituras
         WHERE fonte = 'ana' AND strftime('%s', ts) <= strftime('%s', a.ts_alvo)
         ORDER BY strftime('%s', ts) DESC LIMIT 1) AS ts_antes,
        (SELECT ts FROM leituras
         WHERE fonte = 'ana' AND strftime('%s', ts) >= strftime('%s', a.ts_alvo)
         ORDER BY strftime('%s', ts) ASC LIMIT 1) AS ts_depois
    FROM alvo a
),
casado AS (
    SELECT c.*, o.nivel_cm AS observado_bruto, o.ts AS ts_observado_bruto,
        (o.ts IS NOT NULL
         AND ABS(strftime('%s', o.ts) - strftime('%s', c.ts_alvo)) <= 3600
        ) AS dentro_tolerancia
    FROM cand c
    LEFT JOIN leituras o ON o.fonte = 'ana' AND o.ts = (
        CASE
            WHEN c.ts_antes IS NULL THEN c.ts_depois
            WHEN c.ts_depois IS NULL THEN c.ts_antes
            WHEN ABS(strftime('%s', c.ts_alvo) - strftime('%s', c.ts_antes))
                 <= ABS(strftime('%s', c.ts_depois) - strftime('%s', c.ts_alvo))
            THEN c.ts_antes ELSE c.ts_depois
        END)
)
SELECT ts_ciclo, horizonte, ts_alvo, previsto_sem_chuva, previsto_com_chuva,
       CASE WHEN dentro_tolerancia THEN observado_bruto END AS observado,
       CASE WHEN dentro_tolerancia THEN ts_observado_bruto END AS ts_observado,
       CASE WHEN dentro_tolerancia
            THEN ROUND(previsto_sem_chuva - observado_bruto, 2) END
            AS erro_sem_chuva_cm,
       CASE WHEN dentro_tolerancia
            THEN ROUND(previsto_com_chuva - observado_bruto, 2) END
            AS erro_com_chuva_cm,
       CASE WHEN dentro_tolerancia
            THEN ROUND(nivel_atual - observado_bruto, 2) END
            AS erro_persistencia_cm
FROM casado
"""


def _agora():
    return pd.Timestamp.now(tz=FUSO)


class Banco:
    def __init__(self, caminho=DB_PADRAO):
        self.caminho = caminho
        self.con = sqlite3.connect(caminho)
        self._migrar()
        self.con.executescript(_SCHEMA)
        self.con.execute("DROP VIEW IF EXISTS previsoes_erro")
        self.con.executescript(_VIEW_ERRO)
        self.con.commit()
        self._buffer_mediana = []

    def _migrar(self):
        try:
            tem = self.con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='leituras'").fetchone()
        except sqlite3.Error:
            return
        if not tem:
            return

        try:
            ja_tem_indice = self.con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' "
                "AND name='ix_leituras_ts_fonte'").fetchone()
            if not ja_tem_indice:
                self.con.execute(
                    "DELETE FROM leituras WHERE rowid NOT IN "
                    "(SELECT MAX(rowid) FROM leituras GROUP BY ts, fonte)")
                self.con.commit()
        except sqlite3.Error:
            pass

        try:
            cols = [c[1] for c in
                    self.con.execute("PRAGMA table_info(leituras)").fetchall()]
            if cols and "chuva_mm" not in cols:
                self.con.execute("ALTER TABLE leituras ADD COLUMN chuva_mm REAL")
                self.con.commit()
        except sqlite3.Error:
            pass

        try:
            tem_previsoes = self.con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='previsoes'").fetchone()
            if tem_previsoes:
                cols = [c[1] for c in
                        self.con.execute("PRAGMA table_info(previsoes)").fetchall()]
                novas = {
                    "ana_ultima_leitura": "TEXT",
                    "ana_horas_novas": "INTEGER",
                    "ana_fora_do_ar": "INTEGER",
                    "bacia_horas_previstas": "INTEGER",
                    "bacia_horas_chuva": "INTEGER",
                    "cidade_mmh": "REAL",
                    "cidade_acum6h": "REAL",
                    "estacao_mmh": "REAL",
                    "estacao_acum6h": "REAL",
                    "risco_motivos": "TEXT",
                }
                for nome, tipo in novas.items():
                    if nome not in cols:
                        self.con.execute(
                            f"ALTER TABLE previsoes ADD COLUMN {nome} {tipo}")
                self.con.commit()

                ja_tem_indice_prev = self.con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='index' "
                    "AND name='ix_previsoes_ts_unico'").fetchone()
                if not ja_tem_indice_prev:
                    self.con.execute(
                        "DELETE FROM previsoes WHERE rowid NOT IN "
                        "(SELECT MAX(rowid) FROM previsoes GROUP BY ts)")
                    self.con.commit()
        except sqlite3.Error:
            pass

    def fechar(self):
        self.con.close()

    def gravar_leitura(self, nivel_cm, metodo="", confianca=None,
                       menor_numero=None, ts=None, janela_mediana=3,
                       fonte="camera", chuva_mm=None):
        if nivel_cm is None:
            return None
        self._buffer_mediana.append(float(nivel_cm))
        if len(self._buffer_mediana) > janela_mediana:
            self._buffer_mediana.pop(0)
        valor = float(np.median(self._buffer_mediana))

        ts = ts if ts is not None else _agora()
        self.con.execute(
            "INSERT OR REPLACE INTO leituras (ts, nivel_cm, metodo, confianca, "
            "menor_numero, fonte, chuva_mm) VALUES (?,?,?,?,?,?,?)",
            (pd.Timestamp(ts).isoformat(), valor, metodo, confianca,
             menor_numero, fonte, chuva_mm))
        self.con.commit()
        return valor

    def serie_recente(self, horas=48, fonte=None):
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
        fim = pd.Timestamp(ate).tz_convert(FUSO) if ate is not None else _agora()
        ini = fim - timedelta(hours=horas)
        df = pd.read_sql_query(
            "SELECT ts, nivel_cm, chuva_mm FROM leituras "
            "WHERE ts >= ? AND ts <= ? ORDER BY ts",
            self.con, params=[ini.isoformat(), fim.isoformat()])
        if df.empty:
            return pd.DataFrame(columns=["nivel_cm", "chuva_mm"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(FUSO)
        g = df.set_index("ts").resample("1h")
        serie = pd.DataFrame({
            "nivel_cm": g["nivel_cm"].mean(),
            "chuva_mm": g["chuva_mm"].sum(min_count=1),
        })
        grade = pd.date_range(serie.index.min(), serie.index.max(),
                              freq="1h", tz=FUSO)
        out = serie.reindex(grade)
        out["chuva_mm"] = out["chuva_mm"].fillna(0.0)
        out.index.name = "datahora"
        return out

    def chuva_recente(self, horas=6, fonte="ana"):
        ult = self.con.execute(
            "SELECT MAX(ts) FROM leituras WHERE chuva_mm IS NOT NULL"
        ).fetchone()[0]
        if ult is None:
            return None
        corte = (pd.Timestamp(ult) - timedelta(hours=horas)).isoformat()
        q = ("SELECT ts, chuva_mm FROM leituras "
             "WHERE ts >= ? AND chuva_mm IS NOT NULL")
        args = [corte]
        if fonte:
            q += " AND fonte = ?"
            args.append(fonte)
        df = pd.read_sql_query(q + " ORDER BY ts", self.con, params=args)
        if df.empty:
            return None
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(FUSO)
        fim = df["ts"].max()
        ultima_hora = df[df["ts"] > fim - timedelta(hours=1)]["chuva_mm"].sum()
        return {
            "acum_mm": round(float(df["chuva_mm"].sum()), 1),
            "mmh": round(float(ultima_hora), 1),
            "horas_acum": horas,
            "ts": fim,
            "n_registros": len(df),
        }

    def ultima_leitura(self):
        row = self.con.execute(
            "SELECT ts, nivel_cm, metodo, confianca FROM leituras "
            "ORDER BY ts DESC LIMIT 1").fetchone()
        if not row:
            return None
        return {"ts": pd.Timestamp(row[0]).tz_convert(FUSO),
                "nivel_cm": row[1], "metodo": row[2], "confianca": row[3]}

    def gravar_alerta(self, tipo, estado, nivel_cm, origem, mensagem="", ts=None):
        ts = ts if ts is not None else _agora()
        self.con.execute(
            "INSERT INTO alertas (ts, tipo, estado, nivel_cm, origem, mensagem) "
            "VALUES (?,?,?,?,?,?)",
            (pd.Timestamp(ts).isoformat(), tipo, estado, nivel_cm, origem, mensagem))
        self.con.commit()

    def gravar_previsao(self, prev_sem, prev_com=None, clima=None,
                        risco=None, ts=None, ana_ultima_leitura=None,
                        ana_horas_novas=None, ana_fora_do_ar=None,
                        cidade=None, estacao=None, motivos=None):
        ts_hora = pd.Timestamp(ts if ts is not None else _agora()).floor("h")
        ps = prev_sem or {}
        pc = prev_com or {}
        c = clima or {}
        cid = cidade or {}
        est = estacao or {}
        self.con.execute(
            "INSERT INTO previsoes (ts, nivel_atual, prev_6h, prev_12h, prev_24h,"
            " prev_6h_chuva, prev_12h_chuva, prev_24h_chuva, chuva_total_mm,"
            " chuva_media_mmh, chuva_pico_mmh, chuva_prob_max, risco,"
            " ana_ultima_leitura, ana_horas_novas, ana_fora_do_ar,"
            " bacia_horas_previstas, bacia_horas_chuva,"
            " cidade_mmh, cidade_acum6h, estacao_mmh, estacao_acum6h, risco_motivos)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(ts) DO UPDATE SET"
            " nivel_atual=excluded.nivel_atual, prev_6h=excluded.prev_6h,"
            " prev_12h=excluded.prev_12h, prev_24h=excluded.prev_24h,"
            " prev_6h_chuva=excluded.prev_6h_chuva,"
            " prev_12h_chuva=excluded.prev_12h_chuva,"
            " prev_24h_chuva=excluded.prev_24h_chuva,"
            " chuva_total_mm=excluded.chuva_total_mm,"
            " chuva_media_mmh=excluded.chuva_media_mmh,"
            " chuva_pico_mmh=excluded.chuva_pico_mmh,"
            " chuva_prob_max=excluded.chuva_prob_max, risco=excluded.risco,"
            " ana_ultima_leitura=excluded.ana_ultima_leitura,"
            " ana_horas_novas=excluded.ana_horas_novas,"
            " ana_fora_do_ar=excluded.ana_fora_do_ar,"
            " bacia_horas_previstas=excluded.bacia_horas_previstas,"
            " bacia_horas_chuva=excluded.bacia_horas_chuva,"
            " cidade_mmh=excluded.cidade_mmh, cidade_acum6h=excluded.cidade_acum6h,"
            " estacao_mmh=excluded.estacao_mmh, estacao_acum6h=excluded.estacao_acum6h,"
            " risco_motivos=excluded.risco_motivos",
            (ts_hora.isoformat(),
             ps.get("nivel_atual"),
             ps.get("6h"), ps.get("12h"), ps.get("24h"),
             pc.get("6h"), pc.get("12h"), pc.get("24h"),
             c.get("total_mm"), c.get("media_mmh"), c.get("pico_mmh"),
             c.get("prob_max"), risco,
             (pd.Timestamp(ana_ultima_leitura).isoformat()
              if ana_ultima_leitura is not None else None),
             ana_horas_novas,
             (None if ana_fora_do_ar is None else int(bool(ana_fora_do_ar))),
             c.get("horas_previstas"), c.get("horas_chuva"),
             cid.get("mmh"), cid.get("acum_mm"),
             est.get("mmh"), est.get("acum_mm"),
             ("; ".join(motivos) if motivos else None)))
        self.con.commit()

    def previsoes_recentes(self, horas=72):
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

    def importar_csv_ana(self, caminho_csv, fonte="ana"):
        df = pd.read_csv(caminho_csv, parse_dates=["datahora"])
        df["datahora"] = pd.to_datetime(df["datahora"], utc=True).dt.tz_convert(FUSO)
        df = df.dropna(subset=["nivel_cm"])
        df = df.drop_duplicates(subset=["datahora"], keep="last")

        col_chuva = next((c for c in ("chuva_mm", "chuva", "precipitacao")
                          if c in df.columns), None)
        chuvas = (df[col_chuva].tolist() if col_chuva
                  else [None] * len(df))

        self.con.execute("DELETE FROM leituras WHERE fonte = ?", (fonte,))
        self.con.executemany(
            "INSERT OR REPLACE INTO leituras "
            "(ts, nivel_cm, metodo, fonte, chuva_mm) VALUES (?,?,?,?,?)",
            [(ts.isoformat(), float(n), "estacao_ana", fonte,
              (None if c is None or pd.isna(c) else float(c)))
             for ts, n, c in zip(df["datahora"], df["nivel_cm"], chuvas)])
        self.con.commit()
        return len(df)
