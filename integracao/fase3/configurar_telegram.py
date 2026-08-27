# -*- coding: utf-8 -*-
"""
configurar_telegram.py — HidroVision AI

Descobre o chat_id e testa o envio de alertas pelo Telegram.

Antes de usar:
  1. crie o bot com o @BotFather no Telegram (comando /newbot);
  2. envie qualquer mensagem ao bot recém-criado — sem isso o Telegram não
     revela o chat_id, por segurança;
  3. exporte o token:
         Windows:  $env:TELEGRAM_TOKEN = "seu_token"
         Linux:    export TELEGRAM_TOKEN=seu_token

Uso:
    python configurar_telegram.py --descobrir     # mostra os chat_id disponíveis
    python configurar_telegram.py --testar        # envia mensagem de teste
    python configurar_telegram.py --simular       # envia exemplos de cada alerta
"""
import argparse
import os
import sys

import requests

API = "https://api.telegram.org"


def descobrir(token):
    """Lista as conversas que já falaram com o bot."""
    r = requests.get(f"{API}/bot{token}/getUpdates", timeout=20)
    r.raise_for_status()
    dados = r.json()
    if not dados.get("ok"):
        sys.exit(f"Telegram respondeu erro: {dados}")

    vistos = {}
    for u in dados.get("result", []):
        msg = u.get("message") or u.get("channel_post") or {}
        chat = msg.get("chat") or {}
        if chat.get("id") is not None:
            vistos[chat["id"]] = chat

    if not vistos:
        print("Nenhuma conversa encontrada.\n"
              "Envie uma mensagem ao bot no Telegram e rode de novo.\n"
              "Para um grupo: adicione o bot ao grupo e mande uma mensagem lá.")
        return

    print(f"{len(vistos)} conversa(s) encontrada(s):\n")
    for cid, chat in vistos.items():
        tipo = chat.get("type", "?")
        nome = (chat.get("title")
                or " ".join(filter(None, [chat.get("first_name"),
                                          chat.get("last_name")]))
                or chat.get("username") or "—")
        marca = "  (grupo)" if tipo in ("group", "supergroup") else ""
        print(f"  chat_id: {cid}{marca}")
        print(f"           {nome}  [{tipo}]\n")
    print("Defina o escolhido com:")
    print('  Windows:  $env:TELEGRAM_CHAT_ID = "NUMERO"')
    print("  Linux:    export TELEGRAM_CHAT_ID=NUMERO")


def testar(token, chat_id):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from alertas import CanalTelegram
    CanalTelegram(token, chat_id).testar()


def simular(token, chat_id):
    """Envia um exemplo de cada tipo de alerta, para ver a formatação."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from alertas import CanalTelegram
    c = CanalTelegram(token, chat_id, intervalo_min_seg=0)
    exemplos = [
        "[NÍVEL] ATENÇÃO: água em 40 cm, subindo — limiar de 40 cm atingido "
        "(60 cm até a área urbana)",
        "[TRAJETÓRIA] ALERTA: subindo 18 cm/h, atinge a área urbana em ~2h30",
        "[NÍVEL] EMERGÊNCIA: água em 92 cm, subindo — limiar de 90 cm atingido "
        "(8 cm até a área urbana)",
        "[TRAJETÓRIA] Cancelado (nível estabilizou) — a projeção de tempo até a "
        "área urbana não se aplica",
        "[NÍVEL] Normalizado: água recuou para 60 cm, abaixo do limiar de "
        "alerta (70 cm)",
    ]
    for e in exemplos:
        c.enviar(e)
    print(f"{len(exemplos) - len(c._fila)} mensagem(ns) enviada(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("TELEGRAM_TOKEN"))
    ap.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"))
    ap.add_argument("--descobrir", action="store_true")
    ap.add_argument("--testar", action="store_true")
    ap.add_argument("--simular", action="store_true")
    args = ap.parse_args()

    if not args.token:
        sys.exit("Defina TELEGRAM_TOKEN (ou use --token).")

    if args.descobrir:
        descobrir(args.token)
    elif args.testar or args.simular:
        if not args.chat_id:
            sys.exit("Defina TELEGRAM_CHAT_ID (ou use --chat-id). "
                     "Rode com --descobrir para encontrá-lo.")
        (simular if args.simular else testar)(args.token, args.chat_id)
    else:
        ap.print_help()
