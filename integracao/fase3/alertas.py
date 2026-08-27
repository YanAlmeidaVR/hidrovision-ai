# -*- coding: utf-8 -*-
"""
alertas.py — HidroVision AI (Fase 3)

Dois tipos de alerta INDEPENDENTES, porque descrevem situações diferentes:

1. NÍVEL — a água está a X cm na régua. É um fato, vale em qualquer direção.
   Dispara ao cruzar o limiar para cima; NÃO se desarma só porque começou a
   descer (água em 60 cm recuando ainda é situação anormal). Normaliza quando
   cai abaixo do limiar menos a folga (histerese).

2. TRAJETÓRIA — subindo, atinge o nível crítico em X horas. É uma previsão.
   Dispara quando o tempo projetado entra numa faixa de urgência e CANCELA
   quando a subida cessa (a projeção deixa de valer).

A distinção importa: água a 30 cm subindo 40 cm/h (crítico em 1h45) é mais
urgente que água a 90 cm parada — e um alerta único não distinguiria os casos.
"""
import os
import time
from dataclasses import dataclass, field

import pandas as pd

import projecao as PJ

# ----------------------------------------------------------------------
# limiares de NÍVEL por modo de operação
# ----------------------------------------------------------------------
MODOS = {
    # régua urbana da maquete: 100 cm = a água atinge a área urbana
    "maquete": {"atencao": 40, "alerta": 70, "emergencia": 90,
                "folga": 5, "critico": 100},
    # estação 61305000 (cota do rio) — percentis 90/95/99 do histórico
    "estacao": {"atencao": 228, "alerta": 304, "emergencia": 388,
                "folga": 15, "critico": 447},
}
ORDEM = ("atencao", "alerta", "emergencia")
NOMES = {"atencao": "ATENÇÃO", "alerta": "ALERTA", "emergencia": "EMERGÊNCIA"}


# ----------------------------------------------------------------------
# canais de envio (Telegram entra aqui)
# ----------------------------------------------------------------------
class CanalConsole:
    def enviar(self, mensagem):
        print(f"  >> {mensagem}")


class CanalArquivo:
    def __init__(self, caminho="alertas.log"):
        self.caminho = caminho

    def enviar(self, mensagem):
        with open(self.caminho, "a", encoding="utf-8") as f:
            f.write(f"{pd.Timestamp.now().isoformat()}  {mensagem}\n")


class CanalTelegram:
    """
    Envia os alertas por Telegram.

    Credenciais vêm de variáveis de ambiente (TELEGRAM_TOKEN e
    TELEGRAM_CHAT_ID) para não ficarem no código nem no repositório.

    Duas decisões pensadas para operação em campo:

    1. Falha de rede não derruba o sistema. Se o envio falhar, a mensagem vai
       para uma fila e é reenviada no próximo alerta. Uma enchente costuma
       derrubar energia e conectividade justamente quando o alerta importa, e
       perder a mensagem seria pior do que atrasá-la.
    2. Há um intervalo mínimo entre envios. O gerenciador já aplica histerese,
       mas uma falha inesperada em cascata não pode virar dezenas de
       notificações.
    """

    API = "https://api.telegram.org"

    def __init__(self, token=None, chat_id=None, intervalo_min_seg=20,
                 max_fila=20):
        self.token = token or os.environ.get("TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if not self.token or not self.chat_id:
            raise ValueError(
                "Telegram: defina TELEGRAM_TOKEN e TELEGRAM_CHAT_ID "
                "(ou passe token e chat_id ao construir o canal).")
        self.intervalo_min = intervalo_min_seg
        self.max_fila = max_fila
        self._fila = []
        self._ultimo_envio = 0.0

    # ------------------------------------------------------------------
    def _post(self, texto):
        import requests
        r = requests.post(f"{self.API}/bot{self.token}/sendMessage",
                          json={"chat_id": self.chat_id, "text": texto,
                                "parse_mode": "HTML",
                                "disable_web_page_preview": True},
                          timeout=15)
        r.raise_for_status()
        return r.json()

    def _formatar(self, mensagem):
        """Deixa a mensagem legível no aplicativo."""
        icones = {"EMERGÊNCIA": "\U0001F6A8", "ALERTA": "\u26A0\uFE0F",
                  "ATENÇÃO": "\U0001F4E2", "Normalizado": "\u2705",
                  "Cancelado": "\u2139\uFE0F"}
        icone = next((v for k, v in icones.items() if k in mensagem), "")
        agora = pd.Timestamp.now(tz="America/Sao_Paulo").strftime("%d/%m %H:%M")
        return f"{icone} <b>HidroVision AI</b> · {agora}\n{mensagem}"

    # ------------------------------------------------------------------
    def enviar(self, mensagem):
        self._fila.append(self._formatar(mensagem))
        if len(self._fila) > self.max_fila:      # descarta o mais antigo
            self._fila.pop(0)
        self._drenar()

    def _drenar(self):
        """Tenta enviar o que está na fila, respeitando o intervalo mínimo."""
        if not self._fila:
            return
        espera = self.intervalo_min - (time.time() - self._ultimo_envio)
        if espera > 0:
            return                                # tenta no próximo alerta
        pendentes = list(self._fila)
        for texto in pendentes:
            try:
                self._post(texto)
                self._fila.remove(texto)
                self._ultimo_envio = time.time()
            except Exception as e:
                print(f"[Telegram] envio falhou ({type(e).__name__}); "
                      f"{len(self._fila)} mensagem(ns) na fila para reenvio")
                return

    def testar(self):
        """Envia uma mensagem de teste. Use para validar a configuração."""
        self._post(self._formatar(
            "Canal de alertas configurado. Esta é uma mensagem de teste."))
        print("Telegram: mensagem de teste enviada")


def canais_padrao(verboso=True):
    """
    Console e arquivo sempre; Telegram apenas se as variáveis de ambiente
    estiverem definidas. Assim o sistema roda igual com ou sem o bot
    configurado, sem exigir alteração de código.
    """
    canais = [CanalConsole(), CanalArquivo()]
    if os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        try:
            canais.append(CanalTelegram())
            if verboso:
                print("alertas: canal do Telegram ativo")
        except Exception as e:
            print(f"[aviso] Telegram não configurado ({e})")
    elif verboso:
        print("alertas: Telegram inativo "
              "(defina TELEGRAM_TOKEN e TELEGRAM_CHAT_ID para ativar)")
    return canais


@dataclass
class Evento:
    categoria: str      # 'nivel' | 'trajetoria'
    tipo: str           # atencao | alerta | emergencia
    estado: str         # disparado | normalizado | cancelado | agravado
    valor: float
    mensagem: str

    def __str__(self):
        return self.mensagem


# ----------------------------------------------------------------------
@dataclass
class GerenciadorAlertas:
    modo: str = "maquete"
    canais: list = field(default_factory=lambda: [CanalConsole()])
    banco: object = None
    _nivel_armado: dict = field(default_factory=dict)
    _traj_armada: str | None = None      # urgência atualmente ativa

    # ------------------------------------------------------------------
    def trocar_modo(self, modo):
        if modo not in MODOS:
            raise ValueError(f"modo inválido: {modo} (use {list(MODOS)})")
        self.modo = modo
        self._nivel_armado = {}
        self._traj_armada = None

    @property
    def limiares(self):
        return {k: v for k, v in MODOS[self.modo].items()
                if k not in ("folga", "critico")}

    @property
    def folga(self):
        return MODOS[self.modo]["folga"]

    @property
    def nivel_critico(self):
        return MODOS[self.modo]["critico"]

    # ------------------------------------------------------------------
    def avaliar(self, nivel_cm, tendencia=None, df_recente=None, ts=None):
        """
        Avalia as duas categorias e devolve (eventos, projecao).
        A projeção é retornada para o dashboard exibir o tempo restante.
        """
        eventos = []
        eventos += self._avaliar_nivel(nivel_cm, tendencia, ts)

        proj = PJ.projetar(nivel_cm, tendencia, df_recente,
                           nivel_critico=self.nivel_critico)
        eventos += self._avaliar_trajetoria(proj, ts)
        return eventos, proj

    # ---------------- categoria 1: NÍVEL ----------------
    def _avaliar_nivel(self, nivel, tendencia, ts):
        eventos = []
        if nivel is None:
            return eventos
        direcao = ""
        if tendencia is not None and tendencia.taxa_cm_h is not None:
            direcao = {"subindo": ", subindo", "descendo": ", recuando",
                       "estavel": ", estável"}.get(tendencia.rotulo, "")

        for tipo in ORDEM:
            lim = self.limiares[tipo]
            armado = self._nivel_armado.get(tipo, False)

            if not armado and nivel >= lim:
                self._nivel_armado[tipo] = True
                msg = (f"[NÍVEL] {NOMES[tipo]}: água em {nivel:.0f} cm"
                       f"{direcao} — limiar de {lim} cm atingido "
                       f"({self.nivel_critico - nivel:.0f} cm até a área urbana)")
                self._emitir("nivel", tipo, "disparado", nivel, msg, ts)
                eventos.append(Evento("nivel", tipo, "disparado", nivel, msg))

            elif armado and nivel < lim - self.folga:
                self._nivel_armado[tipo] = False
                msg = (f"[NÍVEL] Normalizado: água recuou para {nivel:.0f} cm, "
                       f"abaixo do limiar de {NOMES[tipo].lower()} ({lim} cm)")
                self._emitir("nivel", tipo, "normalizado", nivel, msg, ts)
                eventos.append(Evento("nivel", tipo, "normalizado", nivel, msg))
        return eventos

    # ---------------- categoria 2: TRAJETÓRIA ----------------
    def _avaliar_trajetoria(self, proj, ts):
        eventos = []
        urg = proj.urgencia if proj.estado == "subindo" else None

        # a subida cessou: a projeção deixa de valer
        if urg is None and self._traj_armada is not None:
            anterior = self._traj_armada
            self._traj_armada = None
            motivo = {"estavel": "nível estabilizou",
                      "descendo": "água começou a recuar",
                      "indefinido": "tendência indefinida"}.get(
                          proj.estado, "trajetória fora das faixas de urgência")
            msg = (f"[TRAJETÓRIA] Cancelado ({motivo}) — a projeção de tempo "
                   f"até a área urbana não se aplica")
            self._emitir("trajetoria", anterior, "cancelado",
                         proj.nivel_cm or 0, msg, ts)
            eventos.append(Evento("trajetoria", anterior, "cancelado",
                                  proj.nivel_cm or 0, msg))
            return eventos

        if urg is None:
            return eventos

        # dispara ou agrava (atenção -> alerta -> emergência)
        if self._traj_armada != urg:
            piorou = (self._traj_armada is not None
                      and ORDEM.index(urg) > ORDEM.index(self._traj_armada))
            estado = "agravado" if piorou else "disparado"
            self._traj_armada = urg
            extra = ""
            if proj.aceleracao_cm_h2 and proj.aceleracao_cm_h2 > 2:
                extra = " — subida ACELERANDO, o tempo real pode ser menor"
            msg = (f"[TRAJETÓRIA] {NOMES[urg]}: subindo "
                   f"{proj.taxa_cm_h:.0f} cm/h, atinge a área urbana em "
                   f"{proj.tempo_formatado}{extra}")
            self._emitir("trajetoria", urg, estado, proj.nivel_cm, msg, ts)
            eventos.append(Evento("trajetoria", urg, estado,
                                  proj.nivel_cm, msg))
        return eventos

    # ------------------------------------------------------------------
    def _emitir(self, categoria, tipo, estado, valor, mensagem, ts):
        for canal in self.canais:
            try:
                canal.enviar(mensagem)
            except Exception as e:
                print(f"[aviso] canal {type(canal).__name__} falhou: {e}")
        if self.banco is not None:
            self.banco.gravar_alerta(tipo, estado, valor, categoria,
                                     mensagem, ts)
