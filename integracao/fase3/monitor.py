# -*- coding: utf-8 -*-
"""
monitor.py — HidroVision AI

Laço de monitoramento contínuo. A cada intervalo (padrão: 1 hora):

  1. consulta o nível ATUAL do rio na estação da ANA (61305000);
  2. consulta a PREVISÃO de chuva para a bacia (Open-Meteo);
  3. grava no banco;
  4. roda os modelos XGBoost em dois cenários:
       - sem chuva adicional  -> o que acontece se não chover mais;
       - com a chuva prevista -> o que acontece se a previsão se confirmar;
  5. combina o resultado com o estado da régua urbana e classifica o risco.

As duas camadas se confirmam:

  RIO (modelo)  — antecipa: "o rio vai subir 80 cm nas próximas 6 h"
  RÉGUA (câmera)— confirma: a água chegou à régua e continua subindo

Rio subindo + água subindo na régua = alta probabilidade de inundação urbana.
"""
import argparse
import os
import time
from dataclasses import dataclass

import pandas as pd

import banco as B
import tendencia as T
import projecao as PJ
import preditor as P
import clima as C
import alertas as A

ESTACAO_ANA = "61305000"
ANA_BASE = "https://www.ana.gov.br/hidrowebservice"


# ----------------------------------------------------------------------
# consulta do nível atual na ANA
# ----------------------------------------------------------------------
class AnaAtual:
    """Cliente mínimo para o nível corrente da estação (token de 60 min)."""

    def __init__(self, identificador, senha):
        self.id, self.senha = identificador, senha
        self.token, self.token_ts = None, 0.0

    def _autenticar(self):
        import requests
        r = requests.get(f"{ANA_BASE}/EstacoesTelemetricas/OAUth/v1",
                         headers={"Identificador": self.id, "Senha": self.senha},
                         timeout=60)
        r.raise_for_status()
        self.token = (r.json().get("items") or {}).get("tokenautenticacao")
        if not self.token:
            raise RuntimeError("ANA: autenticação falhou")
        self.token_ts = time.time()

    def _headers(self):
        if not self.token or time.time() - self.token_ts > 50 * 60:
            self._autenticar()
        return {"Authorization": f"Bearer {self.token}"}

    def leituras_do_dia(self, dia=None, tentativas=3):
        """
        Registros de 15 min do dia (padrão: hoje) -> DataFrame.

        O servidor da ANA devolve 502/503/504 com alguma frequência. Como uma
        falha pode cair justamente no ciclo do evento crítico, a consulta é
        repetida com espera crescente antes de desistir.
        """
        import requests
        dia = dia or pd.Timestamp.now(tz=B.FUSO).date()
        params = {"Código da Estação": ESTACAO_ANA,
                  "Tipo Filtro Data": "DATA_LEITURA",
                  "Data de Busca (yyyy-MM-dd)": dia.isoformat(),
                  "Range Intervalo de busca": "HORA_24"}
        url = (f"{ANA_BASE}/EstacoesTelemetricas/"
               f"HidroinfoanaSerieTelemetricaAdotada/v1")
        r = None
        for n in range(1, tentativas + 1):
            try:
                r = requests.get(url, params=params, headers=self._headers(),
                                 timeout=60)
                if r.status_code == 401:          # token expirou no meio
                    self.token = None
                    continue
                if r.status_code >= 500:
                    if n < tentativas:
                        print(f"  [ANA] HTTP {r.status_code}, tentativa "
                              f"{n}/{tentativas}; nova tentativa em {5*n}s")
                        time.sleep(5 * n)
                        continue
                    r.raise_for_status()
                r.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if n >= tentativas:
                    raise
                print(f"  [ANA] {type(e).__name__}, tentativa {n}/{tentativas}; "
                      f"nova tentativa em {5*n}s")
                time.sleep(5 * n)
        itens = r.json().get("items") or []
        if not itens:
            return pd.DataFrame()
        df = pd.DataFrame(itens)
        df["datahora"] = pd.to_datetime(df["Data_Hora_Medicao"], errors="coerce")
        df["nivel_cm"] = pd.to_numeric(df["Cota_Adotada"], errors="coerce")
        df["chuva_mm"] = pd.to_numeric(df["Chuva_Adotada"], errors="coerce")
        df = df.dropna(subset=["datahora"]).sort_values("datahora")
        df["datahora"] = df["datahora"].dt.tz_localize(
            B.FUSO, nonexistent="shift_forward", ambiguous="NaT")
        return df.dropna(subset=["datahora"])


# ----------------------------------------------------------------------
# classificação de risco combinando rio e régua
# ----------------------------------------------------------------------
@dataclass
class Risco:
    nivel: str            # normal | atencao | alerta | emergencia
    titulo: str
    motivos: list
    rio: dict | None = None
    regua: dict | None = None

    def __str__(self):
        txt = f"[{self.nivel.upper()}] {self.titulo}"
        for m in self.motivos:
            txt += f"\n   - {m}"
        return txt


def avaliar_risco(prev_sem_chuva, prev_com_chuva, resumo_clima,
                  nivel_regua=None, projecao_regua=None, nivel_rio_atual=None):
    """
    Combina a previsão do rio (bacia) com o estado da régua urbana.

    prev_*  : dicts do preditor {'nivel_atual':.., '6h':.., '12h':.., '24h':..}
    nivel_regua / projecao_regua : estado da régua (None se ainda sem água)
    """
    motivos, escalas = [], []

    # ---------- camada 1: o rio ----------
    rio = None
    if prev_sem_chuva:
        atual = prev_sem_chuva["nivel_atual"]
        d6_sem = prev_sem_chuva.get("6h", atual) - atual
        d24_sem = prev_sem_chuva.get("24h", atual) - atual
        d6_com = (prev_com_chuva.get("6h", atual) - atual
                  if prev_com_chuva else d6_sem)
        d24_com = (prev_com_chuva.get("24h", atual) - atual
                   if prev_com_chuva else d24_sem)
        rio = {"atual": atual, "delta6_sem": d6_sem, "delta6_com": d6_com,
               "delta24_sem": d24_sem, "delta24_com": d24_com}

        if d6_com >= 40 or d24_com >= 80:
            escalas.append("alerta")
            motivos.append(f"rio deve subir {d6_com:+.0f} cm em 6 h e "
                           f"{d24_com:+.0f} cm em 24 h")
        elif d6_com >= 15 or d24_com >= 30:
            escalas.append("atencao")
            motivos.append(f"rio em elevação: {d6_com:+.0f} cm previstos em 6 h")
        elif d6_com <= -10:
            motivos.append(f"rio em recessão ({d6_com:+.0f} cm em 6 h)")
        else:
            motivos.append("rio estável na previsão de 6 h")

        # a chuva prevista muda o quadro?
        if prev_com_chuva and (d24_com - d24_sem) >= 15:
            motivos.append(f"a chuva prevista agrava em {d24_com - d24_sem:+.0f} "
                           f"cm o cenário de 24 h")

    if resumo_clima and resumo_clima["total_mm"] >= 30:
        escalas.append("atencao")
        motivos.append(f"previsão de {resumo_clima['total_mm']:.0f} mm "
                       f"em {resumo_clima['horas_previstas']} h")

    # ---------- camada 2: a régua urbana ----------
    regua = None
    if nivel_regua is not None:
        folga = 100 - nivel_regua
        regua = {"nivel": nivel_regua, "folga": folga}
        motivos.append(f"água na régua urbana em {nivel_regua:.0f} cm "
                       f"({folga:.0f} cm até a área urbana)")
        if nivel_regua >= 90:
            escalas.append("emergencia")
        elif nivel_regua >= 70:
            escalas.append("alerta")
        elif nivel_regua >= 40:
            escalas.append("atencao")

        if projecao_regua is not None and projecao_regua.estado == "subindo":
            regua["horas"] = projecao_regua.horas_para_critico
            motivos.append(f"subindo {projecao_regua.taxa_cm_h:.0f} cm/h — "
                           f"atinge a área urbana em "
                           f"{projecao_regua.tempo_formatado}")
            h = projecao_regua.horas_para_critico
            if h is not None:
                if h <= 1:
                    escalas.append("emergencia")
                elif h <= 3:
                    escalas.append("alerta")
                elif h <= 6:
                    escalas.append("atencao")
            # confirmação cruzada: rio subindo E régua subindo
            if rio and rio["delta6_com"] >= 15:
                escalas.append("alerta")
                motivos.append("CONFIRMAÇÃO: modelo prevê rio em elevação e a "
                               "régua confirma a subida")
        elif projecao_regua is not None and projecao_regua.estado == "descendo":
            motivos.append("água recuando na régua")

    ordem = ["normal", "atencao", "alerta", "emergencia"]
    nivel = max(escalas, key=ordem.index) if escalas else "normal"
    titulos = {
        "normal": "Situação normal",
        "atencao": "Atenção — condições em evolução",
        "alerta": "Alerta — risco de inundação urbana",
        "emergencia": "EMERGÊNCIA — água prestes a atingir a área urbana",
    }
    return Risco(nivel, titulos[nivel], motivos, rio, regua)


# ----------------------------------------------------------------------
# ciclo de monitoramento
# ----------------------------------------------------------------------
class Monitor:
    def __init__(self, db="hidrovision.db", pasta_modelos=".",
                 ana_id=None, ana_senha=None, horas_previsao=24,
                 lat=C.LAT_BACIA, lon=C.LON_BACIA, notificar=True,
                 lembrete_horas=6):
        self.banco = B.Banco(db)
        # canais de notificação: console, arquivo e Telegram (se configurado)
        self.canais = A.canais_padrao() if notificar else []
        self._ultimo_risco = None
        # de quantas em quantas horas repetir um alerta que persiste.
        # 0 desativa o lembrete.
        self.lembrete_horas = lembrete_horas
        self._ultima_notificacao = None
        self._inicio_risco = None
        self.preditor = P.Preditor(pasta_modelos)
        self.clima = C.Clima(lat, lon)
        self.horas_previsao = horas_previsao
        self.ana = (AnaAtual(ana_id, ana_senha)
                    if (ana_id and ana_senha) else None)

    # ------------------------------------------------------------------
    def atualizar_rio(self):
        """Busca o nível do dia na ANA e grava as leituras novas."""
        if self.ana is None:
            return 0
        df = self.ana.leituras_do_dia()
        if df.empty:
            return 0
        # agrega para hora (o preditor espera base horária)
        horario = (df.set_index("datahora")
                     .resample("1h")
                     .agg({"nivel_cm": "mean", "chuva_mm": "sum"})
                     .dropna(subset=["nivel_cm"]))
        n = 0
        for ts, row in horario.iterrows():
            self.banco.gravar_leitura(float(row["nivel_cm"]), "estacao_ana",
                                      ts=ts, fonte="ana", janela_mediana=1)
            n += 1
        return n

    def ciclo(self, nivel_regua=None, projecao_regua=None, verboso=True):
        """Um ciclo completo. Devolve (risco, previsoes, resumo_clima)."""
        agora = pd.Timestamp.now(tz=B.FUSO)

        n_novas = 0
        try:
            n_novas = self.atualizar_rio()
        except Exception as e:
            if verboso:
                print(f"  [aviso] ANA indisponível: {e}")

        resumo = None
        try:
            resumo = self.clima.resumo(self.horas_previsao)
        except Exception as e:
            if verboso:
                print(f"  [aviso] previsão do tempo indisponível: {e}")

        serie = self.banco.serie_horaria(horas=30 * 24, ate=agora)
        prev_sem = self.preditor.prever(serie)
        prev_com = None
        if prev_sem and resumo and resumo["media_mmh"] > 0.05:
            prev_com = self.preditor.simular_chuva(
                serie, resumo["media_mmh"],
                horas_de_chuva=min(self.horas_previsao, 24))

        risco = avaliar_risco(prev_sem, prev_com, resumo,
                              nivel_regua, projecao_regua)

        # registra o ciclo para o dashboard montar o histórico
        self.banco.gravar_previsao(prev_sem, prev_com, resumo,
                                   risco.nivel, ts=agora)

        self._notificar(risco, prev_sem, prev_com, resumo, agora)

        if verboso:
            print(f"\n{'='*70}\n{agora:%d/%m/%Y %H:%M}"
                  f"{f'  ({n_novas} leituras novas da ANA)' if n_novas else ''}")
            print(f"  clima: {C.descrever(resumo)}")
            if prev_sem:
                print(f"  rio agora: {prev_sem['nivel_atual']:.0f} cm")
                for h in ("6h", "12h", "24h"):
                    if h in prev_sem:
                        s = prev_sem[h]
                        linha = f"    {h:>4}: {s:6.0f} cm"
                        if prev_com and h in prev_com:
                            linha += f"   |  com a chuva prevista: {prev_com[h]:6.0f} cm"
                        print(linha)
            print(f"\n{risco}")
        return risco, (prev_com or prev_sem), resumo

    def _notificar(self, risco, prev_sem, prev_com, resumo, agora=None):
        """
        Notifica em duas situações:

        1. MUDANÇA de patamar de risco. Um ciclo por hora repetindo "situação
           normal" seria ruído; o que interessa é a transição.
        2. LEMBRETE periódico enquanto um alerta persiste. Um alerta que dura
           doze horas sem nenhuma repetição corre o risco de ser esquecido, e
           a Defesa Civil precisa saber que a condição continua ativa.
        """
        if not self.canais:
            return
        agora = agora or pd.Timestamp.now(tz=B.FUSO)
        mudou = risco.nivel != self._ultimo_risco
        anterior = self._ultimo_risco

        lembrete = False
        if not mudou and risco.nivel != "normal" and self.lembrete_horas:
            if self._ultima_notificacao is not None:
                decorrido = (agora - self._ultima_notificacao).total_seconds() / 3600
                lembrete = decorrido >= self.lembrete_horas

        if not (mudou or lembrete):
            return
        self._ultimo_risco = risco.nivel
        if mudou and anterior is None and risco.nivel == "normal":
            return                      # primeiro ciclo em situação normal
        self._ultima_notificacao = agora

        titulo = {"normal": "Situação normalizada",
                  "atencao": "ATENÇÃO", "alerta": "ALERTA",
                  "emergencia": "EMERGÊNCIA"}[risco.nivel]
        if lembrete:
            horas = ""
            if self._inicio_risco is not None:
                h = (agora - self._inicio_risco).total_seconds() / 3600
                horas = f" há {h:.0f} h"
            cabecalho = f"[RIO] {titulo} — condição ainda ativa{horas}"
        else:
            cabecalho = f"[RIO] {titulo}"
            self._inicio_risco = agora if risco.nivel != "normal" else None

        linhas = [cabecalho]
        linhas += [f"• {m}" for m in risco.motivos]
        if prev_sem:
            linhas.append(f"nível atual: {prev_sem['nivel_atual']:.0f} cm")
            if prev_com and "24h" in prev_com:
                linhas.append(f"previsão 24 h: {prev_sem.get('24h', 0):.0f} cm "
                              f"(com a chuva prevista: {prev_com['24h']:.0f} cm)")
        msg = "\n".join(linhas)
        for canal in self.canais:
            try:
                canal.enviar(msg)
            except Exception as e:
                print(f"[aviso] canal {type(canal).__name__} falhou: {e}")

    def rodar(self, intervalo_min=60, ciclos=None):
        """Laço contínuo. ciclos=None roda indefinidamente."""
        n = 0
        while ciclos is None or n < ciclos:
            try:
                self.ciclo()
            except Exception as e:
                print(f"[erro no ciclo] {e}")
            n += 1
            if ciclos is None or n < ciclos:
                time.sleep(intervalo_min * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Monitoramento contínuo: nível da ANA + previsão de chuva.")
    ap.add_argument("--db", default="hidrovision.db")
    ap.add_argument("--modelos", default=".")
    ap.add_argument("--intervalo", type=int, default=60,
                    help="minutos entre ciclos (padrão 60)")
    ap.add_argument("--ciclos", type=int, default=1,
                    help="quantos ciclos rodar (0 = contínuo)")
    ap.add_argument("--horas-previsao", type=int, default=24)
    ap.add_argument("--lembrete", type=int, default=6,
                    help="repetir alerta ativo a cada N horas (0 desativa)")
    ap.add_argument("--ana-id", default=os.environ.get("ANA_ID"))
    ap.add_argument("--ana-senha", default=os.environ.get("ANA_SENHA"))
    ap.add_argument("--importar", metavar="CSV",
                    help="popula o banco com o histórico antes de começar")
    args = ap.parse_args()

    m = Monitor(args.db, args.modelos, args.ana_id, args.ana_senha,
                args.horas_previsao, lembrete_horas=args.lembrete)
    if args.importar:
        n = m.banco.importar_csv_ana(args.importar)
        print(f"{n} horas históricas importadas de {args.importar}")

    m.rodar(args.intervalo, None if args.ciclos == 0 else args.ciclos)
