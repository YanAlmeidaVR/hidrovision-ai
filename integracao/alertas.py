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


# class CanalTelegram:          # <- plugar quando o bot existir
#     def __init__(self, token, chat_id):
#         self.token, self.chat_id = token, chat_id
#     def enviar(self, mensagem):
#         requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
#                       json={"chat_id": self.chat_id, "text": mensagem},
#                       timeout=10)


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
