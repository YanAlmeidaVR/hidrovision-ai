import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import banco as B
import tendencia as T
import projecao as PJ
import preditor as P
import clima as C
import alertas as A

ESTACAO_ANA = "61305000"
ANA_BASE = "https://www.ana.gov.br/hidrowebservice"

PASTA = Path(__file__).resolve().parent
DB_PADRAO = PASTA / "hidrovision.db"
MODELOS_PADRAO = PASTA.parent / "preditivo" / "modelos"
ENV_PADRAO = PASTA / ".env"
MINUTO_RODADA = 5


def carregar_env(caminho):
    if not caminho.exists():
        return
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


class AnaForaDoAr(Exception):
    pass


class AnaAtual:
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
                if r.status_code == 401:
                    self.token = None
                    continue
                if r.status_code >= 500:
                    if n < tentativas:
                        time.sleep(5 * n)
                        continue
                    raise AnaForaDoAr(f"HTTP {r.status_code}")
                r.raise_for_status()
                break
            except AnaForaDoAr:
                raise
            except requests.exceptions.RequestException:
                if n >= tentativas:
                    raise AnaForaDoAr("sem resposta do servidor")
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


@dataclass
class Risco:
    nivel: str
    titulo: str
    motivos: list
    rio: dict | None = None
    regua: dict | None = None
    local: dict | None = None

    def __str__(self):
        txt = f"[{self.nivel.upper()}] {self.titulo}"
        for m in self.motivos:
            txt += f"\n   - {m}"
        return txt


def avaliar_risco(prev_sem_chuva, prev_com_chuva, resumo_clima,
                  nivel_regua=None, projecao_regua=None, nivel_rio_atual=None,
                  chuva_local=None):
    motivos, escalas = [], []

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

        if prev_com_chuva and (d24_com - d24_sem) >= 15:
            motivos.append(f"a chuva prevista agrava em {d24_com - d24_sem:+.0f} "
                           f"cm o cenário de 24 h")

    if resumo_clima and resumo_clima["total_mm"] >= 30:
        escalas.append("atencao")
        motivos.append(f"previsão de {resumo_clima['total_mm']:.0f} mm "
                       f"em {resumo_clima['horas_previstas']} h")

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
            if rio and rio["delta6_com"] >= 15:
                escalas.append("alerta")
                motivos.append("CONFIRMAÇÃO: modelo prevê rio em elevação e a "
                               "régua confirma a subida")
        elif projecao_regua is not None and projecao_regua.estado == "descendo":
            motivos.append("água recuando na régua")

    local = None
    if chuva_local:
        grau_local, motivo_local = C.avaliar_local(chuva_local)
        local = dict(chuva_local)
        local["grau"] = grau_local
        if motivo_local:
            escalas.append(grau_local)
            motivos.append(motivo_local)
        elif (chuva_local.get("mmh") or 0) >= 0.2:
            motivos.append(C.descrever_local(chuva_local))

    ordem = ["normal", "atencao", "alerta", "emergencia"]
    nivel = max(escalas, key=ordem.index) if escalas else "normal"
    titulos = {
        "normal": "Situação normal",
        "atencao": "Atenção: condições em evolução",
        "alerta": "Alerta: risco de inundação urbana",
        "emergencia": "EMERGÊNCIA: água prestes a atingir a área urbana",
    }
    return Risco(nivel, titulos[nivel], motivos, rio, regua, local)


class Monitor:
    def __init__(self, db=DB_PADRAO, pasta_modelos=MODELOS_PADRAO,
                 ana_id=None, ana_senha=None, horas_previsao=24,
                 lat=C.LAT_BACIA, lon=C.LON_BACIA, notificar=True,
                 lembrete_horas=6, dias_historico=3, minuto_rodada=MINUTO_RODADA):
        self.banco = B.Banco(str(db))
        self.canais = A.canais_padrao() if notificar else []
        self._ultimo_risco = None
        self.lembrete_horas = lembrete_horas
        self._ultima_notificacao = None
        self._inicio_risco = None
        self.preditor = P.Preditor(str(pasta_modelos))
        self.clima = C.Clima(lat, lon)
        self.clima_local = C.Clima(C.LAT_CIDADE, C.LON_CIDADE)
        self.horas_previsao = horas_previsao
        self.ana = (AnaAtual(ana_id, ana_senha)
                    if (ana_id and ana_senha) else None)
        self.dias_historico = dias_historico
        self._ana_fora = False
        self.minuto_rodada = minuto_rodada
        self._ultima_hora_ana = None
        self._ultima_leitura_ana = None

    def atualizar_rio(self, dias=None):
        if self.ana is None:
            return 0
        dias = dias or self.dias_historico
        hoje = pd.Timestamp.now(tz=B.FUSO).date()
        partes = []
        for d in range(dias - 1, -1, -1):
            try:
                df = self.ana.leituras_do_dia(hoje - pd.Timedelta(days=d))
            except AnaForaDoAr:
                if d == 0 and not partes:
                    raise
                continue
            if not df.empty:
                partes.append(df)
        if not partes:
            return 0
        df = pd.concat(partes).sort_values("datahora")
        validas = df.dropna(subset=["nivel_cm"])
        if not validas.empty:
            self._ultima_leitura_ana = validas["datahora"].max()
        horario = (df.set_index("datahora")
                     .resample("1h")
                     .agg({"nivel_cm": "mean", "chuva_mm": "sum"})
                     .dropna(subset=["nivel_cm"]))
        novas = 0
        for ts, row in horario.iterrows():
            chuva = row.get("chuva_mm")
            if chuva is not None and pd.isna(chuva):
                chuva = None
            self.banco.gravar_leitura(float(row["nivel_cm"]), "estacao_ana",
                                      ts=ts, fonte="ana", janela_mediana=1,
                                      chuva_mm=(None if chuva is None
                                                else float(chuva)))
            if self._ultima_hora_ana is None or ts > self._ultima_hora_ana:
                novas += 1
        if not horario.empty:
            ultima = horario.index.max()
            if self._ultima_hora_ana is None or ultima > self._ultima_hora_ana:
                self._ultima_hora_ana = ultima
        return novas

    def ciclo(self, nivel_regua=None, projecao_regua=None, verboso=True):
        agora = pd.Timestamp.now(tz=B.FUSO)
        ana_tinha_baseline = self._ultima_hora_ana is not None

        n_novas = 0
        ana_ok = True
        try:
            n_novas = self.atualizar_rio()
            if self._ana_fora and verboso:
                print("  [ANA] serviço restabelecido")
            self._ana_fora = False
        except AnaForaDoAr:
            ana_ok = False
            if verboso and not self._ana_fora:
                print("  [ANA] SERVIÇO FORA DO AR — ciclo sem leitura de nível")
            self._ana_fora = True
        except Exception as e:
            ana_ok = False
            if verboso:
                print(f"  [ANA] falha na consulta: {e}")

        resumo = None
        try:
            resumo = self.clima.resumo(self.horas_previsao)
        except Exception:
            if verboso:
                print("  [CLIMA] previsão do tempo indisponível neste ciclo")

        local = self.clima_local.agora()
        if local:
            local["origem"] = "Open-Meteo, área urbana"
        cidade_dados = local

        medida = self.banco.chuva_recente(horas=6)
        if medida:
            estacao = {
                "mmh": medida["mmh"],
                "acum_mm": medida["acum_mm"],
                "horas_acum": medida["horas_acum"],
                "intensidade": C.classificar(medida["mmh"]),
                "origem": "pluviômetro da estação",
                "horario": f"{medida['ts']:%d/%m %H:%M}",
            }
            if local is None:
                local = estacao
            else:
                local["estacao"] = estacao

        serie = self.banco.serie_horaria(horas=30 * 24, ate=agora)
        prev_sem = self.preditor.prever(serie)
        prev_com = None
        if prev_sem and resumo and resumo["media_mmh"] > 0.05:
            prev_com = self.preditor.simular_chuva(
                serie, resumo["media_mmh"],
                horas_de_chuva=min(self.horas_previsao, 24))

        risco = avaliar_risco(prev_sem, prev_com, resumo,
                              nivel_regua, projecao_regua,
                              chuva_local=local)

        ana_horas_novas = (None if self.ana is None or not ana_tinha_baseline
                           else n_novas)
        ana_fora_do_ar = None if self.ana is None else (not ana_ok)
        try:
            self.banco.gravar_previsao(
                prev_sem, prev_com, resumo, risco.nivel, ts=agora,
                ana_ultima_leitura=self._ultima_leitura_ana,
                ana_horas_novas=ana_horas_novas,
                ana_fora_do_ar=ana_fora_do_ar,
                cidade=cidade_dados, estacao=medida,
                motivos=risco.motivos)
        except Exception as e:
            print(f"  [BANCO] falha ao gravar o ciclo: {e}")

        self._notificar(risco, prev_sem, prev_com, resumo, agora)

        if verboso:
            if self.ana is None:
                cabecalho_ana = ""
            elif n_novas:
                cabecalho_ana = f"  ({n_novas} h nova(s) da ANA)"
            elif ana_ok:
                cabecalho_ana = "  (nenhuma hora nova da ANA)"
            else:
                cabecalho_ana = ""
            print(f"\n{'='*70}\n{agora:%d/%m/%Y %H:%M}{cabecalho_ana}")
            print(f"  bacia alta (Maria da Fé): {C.descrever(resumo)}")
            origem = (local or {}).get("origem", "sem fonte")
            print(f"  chuva na cidade ({origem}): {C.descrever_local(local)}")
            est = (local or {}).get("estacao")
            if est and abs((est.get("mmh") or 0)
                           - (local.get("mmh") or 0)) >= 0.2:
                print(f"  chuva na estação (pluviômetro): "
                      f"{C.descrever_local(est)}")
            if prev_sem:
                leitura = (f" (leitura ANA de {self._ultima_leitura_ana:%d/%m %H:%M})"
                           if self._ultima_leitura_ana is not None else "")
                print(f"  rio agora: {prev_sem['nivel_atual']:.0f} cm{leitura}")
                for h in ("6h", "12h", "24h"):
                    if h in prev_sem:
                        print(f"    {h:>4}: {prev_sem[h]:6.0f} cm")
            elif not ana_ok:
                print("  rio: sem dado, a fonte de nível está indisponível")
            else:
                motivo = getattr(self.preditor, "motivo", None)
                print(f"  rio: sem previsão ({motivo})" if motivo
                      else "  rio: sem previsão neste ciclo")
            print(f"\n{risco}")
            if not ana_ok:
                print("   ! avaliação feita sem o nível do rio")
        return risco, (prev_com or prev_sem), resumo

    def _notificar(self, risco, prev_sem, prev_com, resumo, agora=None):
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
            return
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

    def _proxima_rodada(self):
        agora = pd.Timestamp.now(tz=B.FUSO)
        alvo = agora.replace(minute=self.minuto_rodada, second=0, microsecond=0)
        if alvo <= agora:
            alvo += pd.Timedelta(hours=1)
        return alvo

    def rodar(self, ciclos=None):
        n = 0
        while ciclos is None or n < ciclos:
            try:
                self.ciclo()
            except Exception as e:
                print(f"[erro no ciclo] {e}")
            n += 1
            if ciclos is not None and n >= ciclos:
                break
            alvo = self._proxima_rodada()
            print(f"\n  próxima rodada às {alvo:%H:%M}")
            while True:
                restante = (alvo - pd.Timestamp.now(tz=B.FUSO)).total_seconds()
                if restante <= 0:
                    break
                time.sleep(min(restante, 60))


if __name__ == "__main__":
    carregar_env(ENV_PADRAO)

    ap = argparse.ArgumentParser(
        description="Monitoramento contínuo: nível da ANA + previsão de chuva.")
    ap.add_argument("--db", default=str(DB_PADRAO))
    ap.add_argument("--modelos", default=str(MODELOS_PADRAO))
    ap.add_argument("--minuto", type=int, default=MINUTO_RODADA,
                    help="minuto de cada hora em que a rodada acontece")
    ap.add_argument("--ciclos", type=int, default=0,
                    help="quantos ciclos rodar (0 = contínuo)")
    ap.add_argument("--horas-previsao", type=int, default=24)
    ap.add_argument("--lembrete", type=int, default=6,
                    help="repetir alerta ativo a cada N horas (0 desativa)")
    ap.add_argument("--ana-id", default=os.environ.get("ANA_ID"))
    ap.add_argument("--ana-senha", default=os.environ.get("ANA_SENHA"))
    ap.add_argument("--dias", type=int, default=3,
                    help="quantos dias de leitura buscar na ANA por ciclo")
    ap.add_argument("--importar", metavar="CSV",
                    help="popula o banco com o histórico antes de começar")
    args = ap.parse_args()

    if not (args.ana_id and args.ana_senha):
        print(f"[aviso] credenciais da ANA não encontradas em {ENV_PADRAO} — "
              f"rodando sem nível do rio")

    m = Monitor(args.db, args.modelos, args.ana_id, args.ana_senha,
                args.horas_previsao, lembrete_horas=args.lembrete,
                dias_historico=args.dias, minuto_rodada=args.minuto)
    if args.importar:
        n = m.banco.importar_csv_ana(args.importar)
        print(f"{n} horas históricas importadas de {args.importar}")

    m.rodar(None if args.ciclos == 0 else args.ciclos)