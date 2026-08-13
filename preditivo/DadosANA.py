# -*- coding: utf-8 -*-
"""
DadosANA.py — HidroVision AI (Fase 2)
Baixa nível/chuva da ANA (HidroWebService) e, opcionalmente, chuva do INMET.

Modos:
  1) Só ANA (gera ana_bruto.csv + ana_horario.csv):
       python DadosANA.py --estacao 61305000 --inicio 2023-01-01 --fim 2026-08-09

  2) ANA + INMET (gera também dados_treino.csv, pronto pro XGBoost):
       python DadosANA.py --estacao 61305000 --inmet A509 --inicio 2023-01-01 --fim 2026-08-09

  3) Descobrir estação INMET mais próxima:
       python DadosANA.py --listar-inmet -22.25 -45.70

Credenciais ANA: variáveis de ambiente ANA_ID e ANA_SENHA
(ou --ana-id / --ana-senha).
"""
import argparse
import math
import os
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

ANA_BASE = "https://www.ana.gov.br/hidrowebservice"
INMET_BASE = "https://apitempo.inmet.gov.br"
TIMEOUT = 60
FUSO = "America/Sao_Paulo"


# ======================================================================
# ANA — HidroWebService (token 60 min, renovação automática)
# ======================================================================
class AnaClient:
    """Cliente autenticado; renova o token sozinho aos 50 min ou em 401."""

    def __init__(self, identificador, senha):
        self.identificador = identificador
        self.senha = senha
        self.token = None
        self.token_ts = 0.0

    def _autenticar(self):
        r = requests.get(f"{ANA_BASE}/EstacoesTelemetricas/OAUth/v1",
                         headers={"Identificador": self.identificador,
                                  "Senha": self.senha},
                         timeout=TIMEOUT)
        r.raise_for_status()
        body = r.json()
        self.token = (body.get("items") or {}).get("tokenautenticacao")
        if not self.token:
            sys.exit(f"ANA: autenticação falhou: {body.get('message')}")
        self.token_ts = time.time()
        print("ANA: token renovado")

    def _headers(self):
        if not self.token or time.time() - self.token_ts > 50 * 60:
            self._autenticar()
        return {"Authorization": f"Bearer {self.token}"}

    def serie_dia(self, cod_estacao, dia):
        """Um dia (HORA_24) da SerieTelemetricaAdotada -> lista de dicts."""
        params = {
            "Código da Estação": cod_estacao,
            "Tipo Filtro Data": "DATA_LEITURA",
            "Data de Busca (yyyy-MM-dd)": dia.isoformat(),
            "Range Intervalo de busca": "HORA_24",
        }
        url = f"{ANA_BASE}/EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada/v1"
        status = "?"
        for tentativa in (1, 2, 3):
            r = requests.get(url, params=params, headers=self._headers(),
                             timeout=TIMEOUT)
            status = r.status_code
            if status == 401:           # token caducou no meio do loop
                self.token = None
                continue
            if status >= 500:           # instabilidade do servidor
                time.sleep(5 * tentativa)
                continue
            r.raise_for_status()
            return r.json().get("items") or []
        print(f"ANA: {dia} falhou após 3 tentativas (HTTP {status}) — pulando")
        return []


def ana_para_horario(items):
    """Registros de 15 min -> DataFrame horário [datahora, nivel_cm, chuva_ana_mm]."""
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df["datahora"] = pd.to_datetime(df["Data_Hora_Medicao"], errors="coerce")
    df["nivel_cm"] = pd.to_numeric(df["Cota_Adotada"], errors="coerce")
    df["chuva_ana_mm"] = pd.to_numeric(df["Chuva_Adotada"], errors="coerce")
    df = (df.dropna(subset=["datahora"])
            [["datahora", "nivel_cm", "chuva_ana_mm"]]
            .sort_values("datahora"))
    df["datahora"] = df["datahora"].dt.tz_localize(
        FUSO, nonexistent="shift_forward", ambiguous="NaT")
    df = df.dropna(subset=["datahora"])
    return (df.set_index("datahora")
              .resample("1h")
              .agg({"nivel_cm": "mean", "chuva_ana_mm": "sum"})
              .reset_index())


def ana_baixar(cli, cod_estacao, inicio, fim, checkpoint="ana_bruto.csv"):
    """Loop dia a dia com checkpoint retomável em CSV."""
    ini = datetime.strptime(inicio, "%Y-%m-%d").date()
    fim_d = datetime.strptime(fim, "%Y-%m-%d").date()

    partes, ja_baixado = [], set()
    if os.path.exists(checkpoint):
        antigo = pd.read_csv(checkpoint)
        antigo["datahora"] = pd.to_datetime(
            antigo["datahora"], utc=True).dt.tz_convert(FUSO)
        partes.append(antigo)
        ja_baixado = set(antigo["datahora"].dt.date.unique())
        print(f"ANA: checkpoint com {len(ja_baixado)} dias — retomando")

    novos, dia = [], ini
    total = (fim_d - ini).days + 1
    feito = 0
    while dia <= fim_d:
        feito += 1
        if dia not in ja_baixado:
            items = cli.serie_dia(cod_estacao, dia)
            if items:
                novos.append(ana_para_horario(items))
            if feito % 30 == 0 and novos:
                pd.concat(partes + novos).drop_duplicates("datahora").to_csv(
                    checkpoint, index=False)
                print(f"ANA: {feito}/{total} dias ({dia})")
            time.sleep(0.3)
        dia += timedelta(days=1)

    if not (partes or novos):
        sys.exit("ANA: nenhum dado no período — confere estação e datas.")
    df = (pd.concat(partes + novos)
            .drop_duplicates("datahora")
            .sort_values("datahora")
            .reset_index(drop=True))
    df.to_csv(checkpoint, index=False)
    return df


def resumo(df, nome):
    print(f"\n{nome}: {len(df)} horas "
          f"({df['datahora'].min()} -> {df['datahora'].max()})")
    if "nivel_cm" in df:
        print(f"cobertura nível: {df['nivel_cm'].notna().mean()*100:.1f}% | "
              f"min {df['nivel_cm'].min():.0f} | max {df['nivel_cm'].max():.0f} | "
              f"média {df['nivel_cm'].mean():.0f} cm")
    if "chuva_mm" in df:
        print(f"cobertura chuva: {df['chuva_mm'].notna().mean()*100:.1f}%")


# ======================================================================
# INMET — API pública (opcional)
# ======================================================================
def inmet_listar_estacoes(lat, lon, n=10):
    r = requests.get(f"{INMET_BASE}/estacoes/T", timeout=TIMEOUT)
    r.raise_for_status()
    ests = r.json()

    def dist(e):
        try:
            return math.hypot(float(e["VL_LATITUDE"]) - lat,
                              float(e["VL_LONGITUDE"]) - lon)
        except (TypeError, ValueError):
            return math.inf
    ests.sort(key=dist)
    return [(e.get("CD_ESTACAO"), e.get("DC_NOME"), e.get("SG_ESTADO"),
             e.get("VL_LATITUDE"), e.get("VL_LONGITUDE")) for e in ests[:n]]


def inmet_baixar_chuva(cod_estacao, inicio, fim):
    """Chuva horária da estação automática (UTC -> America/Sao_Paulo)."""
    blocos = []
    atual = datetime.strptime(inicio, "%Y-%m-%d").date()
    fim_d = datetime.strptime(fim, "%Y-%m-%d").date()
    while atual <= fim_d:
        ate = min(atual + timedelta(days=180), fim_d)
        print(f"INMET {cod_estacao}: {atual} -> {ate}")
        r = requests.get(f"{INMET_BASE}/estacao/{atual}/{ate}/{cod_estacao}",
                         timeout=TIMEOUT)
        if r.status_code != 204 and r.content:
            r.raise_for_status()
            blocos.extend(r.json())
        atual = ate + timedelta(days=1)
        time.sleep(0.5)
    if not blocos:
        sys.exit("INMET: nenhum dado — confere o código da estação.")

    df = pd.DataFrame(blocos)
    col_dt = "DT_MEDICAO" if "DT_MEDICAO" in df.columns else "DTMEDICAO"
    col_hr = "HR_MEDICAO" if "HR_MEDICAO" in df.columns else "HRMEDICAO"
    col_ch = "CHUVA" if "CHUVA" in df.columns else "chuva"
    df["datahora"] = pd.to_datetime(
        df[col_dt].astype(str) + " " + df[col_hr].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    df["chuva_mm"] = pd.to_numeric(df[col_ch], errors="coerce")
    df = (df.dropna(subset=["datahora"])[["datahora", "chuva_mm"]]
            .sort_values("datahora"))
    df["datahora"] = df["datahora"].dt.tz_convert(FUSO)
    return df


# ======================================================================
# Merge -> dataset de treino
# ======================================================================
def montar_dataset(df_ana, df_inmet, saida="dados_treino.csv"):
    df = pd.merge(df_ana, df_inmet, on="datahora", how="left")
    df["chuva_mm"] = df["chuva_mm"].fillna(df["chuva_ana_mm"])
    df = df.drop(columns=["chuva_ana_mm"])
    df = df.set_index("datahora")
    df["nivel_cm"] = df["nivel_cm"].interpolate(limit=3)
    df = df.reset_index()
    df.to_csv(saida, index=False)
    resumo(df, saida)
    return df


# ======================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Baixa dados da ANA (e opcionalmente INMET) para o HidroVision.")
    ap.add_argument("--listar-inmet", nargs=2, type=float, metavar=("LAT", "LON"),
                    help="lista as estações INMET mais próximas e sai")
    ap.add_argument("--estacao", help="código ANA (ex.: 61305000)")
    ap.add_argument("--inmet", help="código INMET (ex.: A509). Opcional: sem ele, baixa só a ANA")
    ap.add_argument("--inicio", default="2023-01-01")
    ap.add_argument("--fim", default=date.today().isoformat())
    ap.add_argument("--saida", default="dados_treino.csv")
    ap.add_argument("--ana-id", default=os.environ.get("ANA_ID"))
    ap.add_argument("--ana-senha", default=os.environ.get("ANA_SENHA"))
    args = ap.parse_args()

    if args.listar_inmet:
        for cod, nome, uf, la, lo in inmet_listar_estacoes(*args.listar_inmet):
            print(f"{cod:>6}  {nome:<30} {uf}  ({la}, {lo})")
        return

    if not args.estacao:
        ap.error("informe --estacao (código da estação ANA)")
    if not (args.ana_id and args.ana_senha):
        ap.error("credenciais da ANA: defina ANA_ID e ANA_SENHA (ou --ana-id/--ana-senha)")

    cli = AnaClient(args.ana_id, args.ana_senha)
    df_ana = ana_baixar(cli, args.estacao, args.inicio, args.fim)
    df_ana.to_csv("ana_horario.csv", index=False)
    resumo(df_ana, "ana_horario.csv")

    if args.inmet:
        df_inmet = inmet_baixar_chuva(args.inmet, args.inicio, args.fim)
        montar_dataset(df_ana, df_inmet, args.saida)
    else:
        print("\n(sem --inmet: baixado só a ANA. Rode de novo com --inmet CODIGO "
              "para gerar o dados_treino.csv — o checkpoint evita rebaixar.)")


if __name__ == "__main__":
    main()