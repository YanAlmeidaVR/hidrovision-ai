# -*- coding: utf-8 -*-
"""
servidor_camera.py — HidroVision AI
Roda na Raspberry: lê a régua com a câmera e publica o resultado na rede local
para o dashboard do notebook.

    GET /video     vídeo ao vivo com as detecções desenhadas (MJPEG, abre em <img>)
    GET /leitura   última leitura da régua em JSON
    GET /          página simples para conferir no navegador

Uso:
    python servidor_camera.py --modelo hidrovision_v06_regua_w8a32.tflite --webcam 0
    no notebook: http://raspberrypi.local:8000
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from ultralytics import YOLO

import mdYOLO as V

ESTADO = {}


def leitura_json():
    r = ESTADO["detector"].ler()
    if r is None:
        return {"pronto": False}
    leitura = r["leitura"]
    return {
        "pronto": True,
        "ts": time.time(),
        "nivel_cm": r["suave"],
        "nivel_bruto_cm": leitura.nivel_cm,
        "metodo": leitura.metodo,
        "confianca": leitura.confianca,
        "detalhe": leitura.detalhe,
        "numeros": sorted({d.valor for d in r["numeros"]}),
        "faltando": list(leitura.lacunas),
        "ms": r["ms"],
    }


def quadro_jpeg(qualidade):
    quadro = ESTADO["camera"].ler()
    if quadro is None:
        return None
    img = V.quadro_anotado(quadro, ESTADO["detector"].ler())
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    return buf.tobytes() if ok else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _cabecalhos(self, tipo, extra=None):
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/leitura"):
            corpo = json.dumps(leitura_json(), ensure_ascii=False).encode("utf-8")
            self._cabecalhos("application/json; charset=utf-8")
            self.wfile.write(corpo)
            return
        if self.path.startswith("/video"):
            self._cabecalhos("multipart/x-mixed-replace; boundary=quadro")
            intervalo = 1 / ESTADO["fps"]
            try:
                while True:
                    jpg = quadro_jpeg(ESTADO["qualidade"])
                    if jpg is not None:
                        self.wfile.write(b"--quadro\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                                         + jpg + b"\r\n")
                    time.sleep(intervalo)
            except (BrokenPipeError, ConnectionResetError):
                return
        if self.path in ("/", "/index.html"):
            html = ("<html><body style='margin:0;background:#111;color:#eee;font-family:sans-serif'>"
                    "<img src='/video' style='width:100%'>"
                    "<pre id='l' style='padding:10px'></pre><script>"
                    "setInterval(async () => { const r = await fetch('/leitura'); "
                    "document.getElementById('l').textContent = JSON.stringify(await r.json(), null, 2); }, 1000);"
                    "</script></body></html>").encode("utf-8")
            self._cabecalhos("text/html; charset=utf-8")
            self.wfile.write(html)
            return
        self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="hidrovision_v06_regua_w8a32.tflite")
    ap.add_argument("--webcam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--porta", type=int, default=8000)
    ap.add_argument("--fps", type=float, default=12.0, help="quadros por segundo enviados no vídeo")
    ap.add_argument("--qualidade", type=int, default=70, help="qualidade do JPEG, de 1 a 100")
    args = ap.parse_args()

    modelo = YOLO(args.modelo)
    V.montar_paleta(modelo.names)
    print(f"modelo: {args.modelo} | classes: {len(modelo.names)}")

    cap = V.abrir_camera(args.webcam)
    if cap is None:
        print("não foi possível abrir a câmera", args.webcam)
        return

    ESTADO["camera"] = V.CameraAoVivo(cap)
    ESTADO["detector"] = V.DetectorAoVivo(modelo, ESTADO["camera"], args.imgsz)
    ESTADO["fps"] = args.fps
    ESTADO["qualidade"] = args.qualidade

    servidor = ThreadingHTTPServer(("0.0.0.0", args.porta), Handler)
    servidor.daemon_threads = True
    print(f"servindo em http://0.0.0.0:{args.porta}  (vídeo em /video, leitura em /leitura)")
    print("Ctrl+C para sair")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ESTADO["detector"].parar()
        ESTADO["camera"].parar()
        servidor.server_close()


if __name__ == "__main__":
    main()
