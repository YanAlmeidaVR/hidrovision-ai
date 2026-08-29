# -*- coding: utf-8 -*-
"""
clima.py — HidroVision AI

Chuva pela API do Open-Meteo (gratuita, sem chave de acesso).

Duas leituras, com papéis diferentes, porque a água chega de dois jeitos:

  BACIA ALTA (Maria da Fé, a montante)
      É a chuva que ainda vai escoar até o trecho monitorado, com horas de
      atraso. Alimenta os modelos do rio, e é a única com que eles foram
      treinados. Trocar essa fonte invalidaria a previsão.

  CIDADE (Santa Rita do Sapucaí)
      É a chuva que cai aqui e vai direto para a drenagem urbana. Não passa
      pelo modelo do rio: alagamento de rua não espera a cheia chegar. Serve
      de alerta local imediato, com regra própria.

Quem opera precisa das duas na tela. Ver "1 mm previsto" durante um temporal
derruba a confiança no painel, mesmo com o número certo: ele responde sobre
outro lugar e outro horizonte.
"""
import requests

API = "https://api.open-meteo.com/v1/forecast"
FUSO = "America/Sao_Paulo"
TIMEOUT = 30

# bacia alta do Sapucaí (região de Maria da Fé / Mantiqueira)
LAT_BACIA, LON_BACIA = -22.31, -45.37
# a própria cidade
LAT_CIDADE, LON_CIDADE = -22.25, -45.70

# Classificação usual de intensidade, em mm/h — a mesma escala que o
# meteorologista usa. O que alaga rua é intensidade, não o acumulado do mês.
FAIXAS = ((50.0, "muito forte"), (25.0, "forte"),
          (5.0, "moderada"), (0.2, "fraca"))

# Limiares do alerta local. Intensidade alta alaga rápido; acumulado alto
# satura solo e drenagem, e alaga mesmo com intensidade menor. Os dois contam.
LOCAL_ATENCAO_MMH = 25.0
LOCAL_ALERTA_MMH = 50.0
LOCAL_ATENCAO_ACUM = 30.0     # mm nas últimas horas
LOCAL_ALERTA_ACUM = 60.0


def classificar(mmh):
    """Nome da faixa de intensidade a partir de mm/h."""
    if mmh is None:
        return "desconhecida"
    for limite, nome in FAIXAS:
        if mmh >= limite:
            return nome
    return "sem chuva"


class ClimaFixo:
    """
    Previsão congelada em arquivo JSON, para apresentações.

    Numa feira, depender de rede é risco desnecessário. Esta classe expõe a
    mesma interface da Clima, lendo de um arquivo preparado antes.

    Formato esperado do JSON:
        {"horas_previstas": 72, "total_mm": 120.0, "media_mmh": 1.7,
         "pico_mmh": 18.0, "prob_max": 95, "horas_chuva": 34,
         "temp_atual": 19.0, "descricao": "texto opcional",
         "local": {"mmh": 45.0, "acum_mm": 70.0, "horas_acum": 6}}
    """

    def __init__(self, caminho):
        import json
        with open(caminho, encoding="utf-8") as f:
            self._dados = json.load(f)

    def resumo(self, horas=None):
        return dict(self._dados)

    def previsao(self, horas=None):
        return self._dados.get("horario", {})

    def agora(self, horas_passadas=6):
        local = self._dados.get("local")
        if not local:
            return None
        d = dict(local)
        d.setdefault("horas_acum", horas_passadas)
        d.setdefault("intensidade", classificar(d.get("mmh")))
        return d


class Clima:
    def __init__(self, lat=LAT_BACIA, lon=LON_BACIA):
        self.lat, self.lon = lat, lon

    # ------------------------------------------------------------------
    # previsão (alimenta os modelos do rio)
    # ------------------------------------------------------------------
    def previsao(self, horas=24):
        """
        Previsão horária:
          {'horas': [ts...], 'chuva_mm': [...], 'prob': [...], 'temp': [...]}
        Levanta exceção se a API falhar — o chamador decide o que fazer.
        """
        r = requests.get(API, params={
            "latitude": self.lat, "longitude": self.lon,
            "hourly": "precipitation,precipitation_probability,temperature_2m",
            "timezone": FUSO,
            "forecast_hours": max(1, min(int(horas), 168)),
        }, timeout=TIMEOUT)
        r.raise_for_status()
        h = r.json().get("hourly", {})
        return {
            "horas": h.get("time", []),
            "chuva_mm": [x or 0.0 for x in h.get("precipitation", [])],
            "prob": [x or 0 for x in h.get("precipitation_probability", [])],
            "temp": h.get("temperature_2m", []),
        }

    def resumo(self, horas=24):
        """
        Condensa a previsão nos números que interessam ao preditor:
          total_mm    chuva acumulada prevista no período
          media_mmh   intensidade média, injetada nas features
          pico_mmh    maior intensidade horária prevista
          prob_max    maior probabilidade de precipitação
          horas_chuva quantas horas com chuva acima de 0,1 mm
        """
        p = self.previsao(horas)
        chuva = p["chuva_mm"]
        if not chuva:
            return None
        total = float(sum(chuva))
        return {
            "horas_previstas": len(chuva),
            "total_mm": round(total, 1),
            "media_mmh": round(total / len(chuva), 2),
            "pico_mmh": round(float(max(chuva)), 1),
            "prob_max": int(max(p["prob"]) if p["prob"] else 0),
            "horas_chuva": sum(1 for c in chuva if c > 0.1),
            "temp_atual": (round(float(p["temp"][0]), 1)
                           if p.get("temp") else None),
        }

    # ------------------------------------------------------------------
    # condição observada agora (alimenta o alerta local)
    # ------------------------------------------------------------------
    def agora(self, horas_passadas=6):
        """
        Chuva caindo agora e acumulada nas últimas horas.

        Devolve dict com mmh, acum_mm, horas_acum, intensidade, temp e
        horario. Devolve None se a API não responder: faltar a condição local
        não pode derrubar o ciclo do rio.
        """
        try:
            r = requests.get(API, params={
                "latitude": self.lat, "longitude": self.lon,
                "current": "precipitation,rain,temperature_2m",
                "hourly": "precipitation",
                "past_hours": max(1, min(int(horas_passadas), 24)),
                "forecast_hours": 1,
                "timezone": FUSO,
            }, timeout=TIMEOUT)
            r.raise_for_status()
            d = r.json()
        except Exception:
            return None

        atual = d.get("current") or {}
        mmh = atual.get("precipitation")
        if mmh is None:
            mmh = atual.get("rain")

        horaria = (d.get("hourly") or {}).get("precipitation") or []
        passadas = [x or 0.0 for x in horaria[:horas_passadas]]
        acum = round(float(sum(passadas)), 1) if passadas else 0.0

        # sem o bloco 'current', usa a última hora fechada
        if mmh is None and passadas:
            mmh = passadas[-1]
        mmh = round(float(mmh), 1) if mmh is not None else 0.0

        return {
            "mmh": mmh,
            "acum_mm": acum,
            "horas_acum": horas_passadas,
            "intensidade": classificar(mmh),
            "temp": atual.get("temperature_2m"),
            "horario": atual.get("time"),
        }


def avaliar_local(local):
    """
    Risco de alagamento urbano a partir da chuva na cidade.

    Regra direta sobre intensidade e acumulado, sem passar pelos modelos do
    rio: a água que cai aqui vai para a drenagem, não para o leito
    monitorado, e alaga antes de o rio subir.

    Devolve (nivel, motivo), com nivel em normal | atencao | alerta.
    """
    if not local:
        return "normal", None
    mmh = local.get("mmh") or 0.0
    acum = local.get("acum_mm") or 0.0
    h = local.get("horas_acum", 6)
    nome = local.get("intensidade", classificar(mmh))

    if mmh >= LOCAL_ALERTA_MMH or acum >= LOCAL_ALERTA_ACUM:
        return "alerta", (f"chuva {nome} na cidade: {mmh:.0f} mm/h agora, "
                          f"{acum:.0f} mm nas últimas {h} h — risco de "
                          f"alagamento urbano")
    if mmh >= LOCAL_ATENCAO_MMH or acum >= LOCAL_ATENCAO_ACUM:
        return "atencao", (f"chuva {nome} na cidade: {mmh:.0f} mm/h agora, "
                           f"{acum:.0f} mm nas últimas {h} h")
    return "normal", None


def descrever(resumo):
    """Frase legível da previsão da bacia."""
    if not resumo:
        return "previsão indisponível"
    if resumo.get("descricao"):
        return resumo["descricao"]
    if resumo["total_mm"] < 1:
        return (f"sem chuva significativa prevista nas próximas "
                f"{resumo['horas_previstas']} h")
    return (f"{resumo['total_mm']:.0f} mm previstos em "
            f"{resumo['horas_previstas']} h "
            f"({resumo['horas_chuva']} h com chuva, pico de "
            f"{resumo['pico_mmh']:.0f} mm/h, prob. máx. {resumo['prob_max']}%)")


def descrever_local(local):
    """Frase legível da condição observada na cidade."""
    if not local:
        return "condição local indisponível"
    mmh = local.get("mmh") or 0.0
    acum = local.get("acum_mm") or 0.0
    h = local.get("horas_acum", 6)
    if mmh >= 0.2:
        return (f"chuva {local['intensidade']} agora: {mmh:.1f} mm/h "
                f"({acum:.0f} mm nas últimas {h} h)")
    if acum >= 0.5:
        return f"sem chuva agora, {acum:.0f} mm nas últimas {h} h"
    return "sem chuva na cidade"