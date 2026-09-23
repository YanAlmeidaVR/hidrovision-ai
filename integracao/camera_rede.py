# -*- coding: utf-8 -*-
"""
Acesso à câmera da Raspberry (servidor_camera.py) pela rede.

Usado pelo dashboard principal e pela página "Régua ao vivo". Não importa o
YOLO nem o OpenCV: o vídeo é aberto pelo navegador direto da Raspberry, e aqui
só se consulta o JSON do /leitura.
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDERECO_PADRAO = "http://raspberrypi.local:8000"

# a resolução de nome (raspberrypi.local) não respeita o timeout do urlopen e
# pode levar vários segundos quando a Raspberry não está na rede; a consulta
# roda numa thread e quem chama espera no máximo o prazo pedido
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="camera_rede")


def _baixar_leitura(url, timeout):
    with urllib.request.urlopen(f"{url}/leitura", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def buscar_leitura(url, timeout=3):
    """Última leitura publicada pela Raspberry. Levanta exceção se não responder
    dentro do prazo."""
    tarefa = _EXECUTOR.submit(_baixar_leitura, url, timeout)
    try:
        return tarefa.result(timeout=timeout)
    finally:
        # se ainda estava na fila, não chega a rodar; evita acumular consultas
        # enquanto a Raspberry está fora
        tarefa.cancel()


def html_video(url, estilo="width:100%;border-radius:10px"):
    return f'<img src="{url}/video" style="{estilo}">'
