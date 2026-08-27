# -*- coding: utf-8 -*-
"""
clima.py — HidroVision AI

Previsão de chuva pela API do Open-Meteo (gratuita, sem chave de acesso).

O modelo preditivo aprendeu com a chuva que JÁ CAIU. Para responder "e se chover
nas próximas horas?", a chuva prevista é injetada nas variáveis de chuva
acumulada como se estivesse caindo agora — o mesmo mecanismo da simulação
manual, porém alimentado por um serviço meteorológico em vez de um valor
arbitrário.

Coordenadas padrão: bacia alta do Sapucaí, a montante de Santa Rita — é a chuva
que ainda vai escoar para o trecho monitorado.
"""
import requests

API = "https://api.open-meteo.com/v1/forecast"
FUSO = "America/Sao_Paulo"
TIMEOUT = 30

# bacia alta do Sapucaí (região de Maria da Fé / Mantiqueira)
LAT_BACIA, LON_BACIA = -22.31, -45.37
# a própria cidade
LAT_CIDADE, LON_CIDADE = -22.25, -45.70


class ClimaFixo:
    """
    Previsão congelada em arquivo JSON, para apresentações.

    Numa feira, depender de rede é risco desnecessário: se o Wi-Fi falhar, a
    demonstração perde a camada meteorológica. Esta classe expõe a mesma
    interface da Clima, lendo de um arquivo preparado antes.

    Formato esperado do JSON:
        {"horas_previstas": 72, "total_mm": 120.0, "media_mmh": 1.7,
         "pico_mmh": 18.0, "prob_max": 95, "horas_chuva": 34,
         "temp_atual": 19.0, "descricao": "texto opcional"}
    """

    def __init__(self, caminho):
        import json
        with open(caminho, encoding="utf-8") as f:
            self._dados = json.load(f)

    def resumo(self, horas=None):
        return dict(self._dados)

    def previsao(self, horas=None):
        return self._dados.get("horario", {})


class Clima:
    def __init__(self, lat=LAT_BACIA, lon=LON_BACIA):
        self.lat, self.lon = lat, lon

    def previsao(self, horas=24):
        """
        Retorna dict com a previsão horária:
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
          total_mm      — chuva acumulada prevista no período
          media_mmh     — intensidade média (mm/h) para injetar nas features
          pico_mmh      — maior intensidade horária prevista
          prob_max      — maior probabilidade de precipitação
          horas_chuva   — quantas horas com chuva acima de 0,1 mm
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


def descrever(resumo):
    """Frase legível a partir do resumo da previsão."""
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
