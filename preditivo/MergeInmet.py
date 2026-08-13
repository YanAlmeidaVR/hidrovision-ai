# -*- coding: utf-8 -*-
"""
MergeINMET.py — HidroVision AI (Fase 2, plano B do INMET)
Lê os pacotes anuais de dados históricos do INMET (ZIPs de
https://portal.inmet.gov.br/dadoshistoricos), extrai a chuva horária da
estação escolhida, converte UTC -> America/Sao_Paulo e cruza com o
ana_horario.csv para gerar o dados_treino.csv.

Uso:
  1) Baixe os ZIPs dos anos desejados (2023.zip, 2024.zip, 2025.zip, 2026.zip)
     em https://portal.inmet.gov.br/dadoshistoricos e coloque nesta pasta
     (não precisa descompactar).
  2) python MergeINMET.py --estacao-inmet A531
     (roda na pasta onde estão os ZIPs e o ana_horario.csv)

Também aceita CSVs já extraídos soltos na pasta.
"""
import argparse
import glob
import io
import os
import sys
import zipfile

import pandas as pd

FUSO = "America/Sao_Paulo"


def _achar_arquivos(cod_estacao, pasta="."):
    """Encontra CSVs da estação dentro de ZIPs anuais ou soltos na pasta."""
    achados = []  # (origem, nome, bytes)
    for z in sorted(glob.glob(os.path.join(pasta, "*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for nome in zf.namelist():
                if cod_estacao.upper() in nome.upper() and nome.upper().endswith(".CSV"):
                    achados.append((z, nome, zf.read(nome)))
    for c in sorted(glob.glob(os.path.join(pasta, "*.CSV")) +
                    glob.glob(os.path.join(pasta, "*.csv"))):
        base = os.path.basename(c).upper()
        if cod_estacao.upper() in base and base.startswith("INMET"):
            achados.append((c, os.path.basename(c), open(c, "rb").read()))
    return achados


def _ler_csv_inmet(raw):
    """CSV histórico do INMET -> DataFrame [datahora(SP), chuva_mm]."""
    texto = raw.decode("latin-1", errors="replace")
    linhas = texto.splitlines()
    # o cabeçalho verdadeiro é a linha que começa com 'Data;'
    idx = next(i for i, ln in enumerate(linhas) if ln.upper().startswith("DATA;"))
    df = pd.read_csv(io.StringIO("\n".join(linhas[idx:])), sep=";", decimal=",",
                     dtype=str)
    df.columns = [c.strip() for c in df.columns]

    col_data = next(c for c in df.columns if c.upper().startswith("DATA"))
    col_hora = next(c for c in df.columns if "HORA" in c.upper() and "UTC" in c.upper())
    col_prec = next(c for c in df.columns if c.upper().startswith("PRECIPITA"))

    hora = df[col_hora].str.replace("UTC", "", regex=False).str.strip().str.zfill(4)
    data = df[col_data].str.strip()
    # formatos: '2023/01/01' (novo) ou '01/01/2023' (antigo)
    dayfirst = data.str.match(r"^\d{2}/\d{2}/\d{4}$").iloc[0]
    df["datahora"] = pd.to_datetime(data + " " + hora, dayfirst=bool(dayfirst),
                                    errors="coerce", utc=True,
                                    format="%d/%m/%Y %H%M" if dayfirst
                                    else "%Y/%m/%d %H%M")
    chuva = pd.to_numeric(df[col_prec].str.replace(",", ".", regex=False),
                          errors="coerce")
    chuva = chuva.where(chuva >= 0)          # -9999 = sem dado
    out = pd.DataFrame({"datahora": df["datahora"], "chuva_mm": chuva})
    out = out.dropna(subset=["datahora"]).sort_values("datahora")
    out["datahora"] = out["datahora"].dt.tz_convert(FUSO)
    return out


def carregar_inmet(cod_estacao, pasta="."):
    achados = _achar_arquivos(cod_estacao, pasta)
    if not achados:
        sys.exit(f"Nenhum CSV da estação {cod_estacao} encontrado nos ZIPs/CSVs "
                 f"desta pasta. Baixe os anos em portal.inmet.gov.br/dadoshistoricos.")
    partes = []
    for origem, nome, raw in achados:
        print(f"lendo {nome}  (de {os.path.basename(origem)})")
        partes.append(_ler_csv_inmet(raw))
    df = (pd.concat(partes)
            .drop_duplicates("datahora")
            .sort_values("datahora")
            .reset_index(drop=True))
    return df


def montar_dataset(df_ana, df_inmet, saida="dados_treino.csv"):
    df = pd.merge(df_ana, df_inmet, on="datahora", how="left")
    if "chuva_ana_mm" in df.columns:
        df["chuva_mm"] = df["chuva_mm"].fillna(df["chuva_ana_mm"])
        df = df.drop(columns=["chuva_ana_mm"])
    df = df.set_index("datahora")
    df["nivel_cm"] = df["nivel_cm"].interpolate(limit=3)
    df = df.reset_index()
    df.to_csv(saida, index=False)
    print(f"\n{saida}: {len(df)} horas "
          f"({df['datahora'].min()} -> {df['datahora'].max()})")
    print(f"cobertura nível: {df['nivel_cm'].notna().mean()*100:.1f}% | "
          f"chuva: {df['chuva_mm'].notna().mean()*100:.1f}%")
    print(f"nível: min {df['nivel_cm'].min():.0f} | max {df['nivel_cm'].max():.0f} | "
          f"média {df['nivel_cm'].mean():.0f} cm | "
          f"chuva total {df['chuva_mm'].sum():.0f} mm")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estacao-inmet", required=True, help="ex.: A531")
    ap.add_argument("--ana", default="ana_horario.csv")
    ap.add_argument("--saida", default="dados_treino.csv")
    ap.add_argument("--pasta", default=".")
    args = ap.parse_args()

    if not os.path.exists(args.ana):
        sys.exit(f"{args.ana} não encontrado — rode na pasta dos dados da ANA.")
    df_ana = pd.read_csv(args.ana)
    df_ana["datahora"] = pd.to_datetime(df_ana["datahora"], utc=True).dt.tz_convert(FUSO)

    df_inmet = carregar_inmet(args.estacao_inmet, args.pasta)
    print(f"\nINMET {args.estacao_inmet}: {len(df_inmet)} horas "
          f"({df_inmet['datahora'].min()} -> {df_inmet['datahora'].max()}) | "
          f"cobertura {df_inmet['chuva_mm'].notna().mean()*100:.1f}%")

    montar_dataset(df_ana, df_inmet, args.saida)


if __name__ == "__main__":
    main()