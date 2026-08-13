# -*- coding: utf-8 -*-
"""
geometria.py — HidroVision AI
Validação e correção geométrica das detecções na régua linimétrica.

Ideia central: os números da régua não estão em posições arbitrárias. Eles são
monotônicos e igualmente espaçados. Isso permite usar a POSIÇÃO para corrigir
o RÓTULO quando o detector erra a classe, e para descobrir números que o
detector deixou passar (miss).

Algoritmo:
  1. Ajusta a reta valor(cm) -> posição(px) por consenso robusto (RANSAC
     simplificado): testa pares de detecções e escolhe a reta com mais
     detecções concordantes, ponderando pela confiança.
  2. Reclassifica quem divergir da reta (valor esperado pela posição), desde
     que a confiança seja baixa — detecção confiante e coerente é preservada.
  3. Remove duplicatas do mesmo valor mantendo a de maior confiança.
  4. Detecta lacunas no espaçamento: números que deveriam existir e não foram
     detectados (miss), reportados para a leitura de nível.
"""
from dataclasses import dataclass, replace

import numpy as np

PASSO_CM = 10           # régua com marcações de 10 em 10 cm
TOL_CM = 4.0            # divergência tolerada entre rótulo e posição
CONF_INTOCAVEL = 0.75   # acima disso, só corrige com evidência muito forte
MIN_INLIERS = 3         # mínimo de detecções concordantes para confiar na reta


@dataclass
class Ajuste:
    """Reta valor -> y: y = a * valor + b (a negativo: valor maior, y menor)."""
    a: float
    b: float
    inliers: list          # detecções concordantes
    r2: float

    @property
    def px_por_cm(self):
        return abs(self.a)

    def valor_em(self, y):
        """Valor em cm correspondente a uma posição vertical."""
        return (y - self.b) / self.a

    def y_de(self, valor):
        return self.a * valor + self.b


def _r2(valores, ys, a, b):
    pred = a * np.asarray(valores) + b
    ys = np.asarray(ys)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def ajustar_robusto(dets, tol_cm=TOL_CM):
    """
    Encontra a reta valor->posição com maior consenso.
    Testa todos os pares (dets é pequeno: no máximo ~11 números).
    Retorna Ajuste ou None.
    """
    if len(dets) < 2:
        return None

    melhor = None
    melhor_score = -1.0
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            d1, d2 = dets[i], dets[j]
            if d1.valor == d2.valor:
                continue
            a = (d2.cy - d1.cy) / (d2.valor - d1.valor)
            if a >= 0:                       # valor maior tem que ficar acima
                continue
            b = d1.cy - a * d1.valor
            px_cm = abs(a)
            if px_cm < 0.5:                  # escala implausível
                continue
            tol_px = tol_cm * px_cm
            inliers = [d for d in dets if abs(d.cy - (a * d.valor + b)) <= tol_px]
            if len(inliers) < 2:
                continue
            # score: nº de concordantes + soma das confianças (desempate)
            score = len(inliers) + sum(d.conf for d in inliers) / 10.0
            if score > melhor_score:
                melhor_score = score
                melhor = (a, b, inliers)

    if melhor is None:
        return None
    a, b, inliers = melhor
    # refina a reta com todos os inliers (mínimos quadrados ponderado por conf)
    if len(inliers) >= 2:
        vals = np.array([d.valor for d in inliers], dtype=float)
        ys = np.array([d.cy for d in inliers], dtype=float)
        w = np.array([d.conf for d in inliers], dtype=float)
        if len(np.unique(vals)) >= 2:
            a2, b2 = np.polyfit(vals, ys, 1, w=w)
            if a2 < 0 and abs(a2) >= 0.5:
                a, b = float(a2), float(b2)
    vals = [d.valor for d in inliers]
    ys = [d.cy for d in inliers]
    return Ajuste(a=a, b=b, inliers=inliers, r2=_r2(vals, ys, a, b))


def _arredondar_passo(valor, passo=PASSO_CM):
    return int(round(valor / passo) * passo)


def corrigir(dets, tol_cm=TOL_CM, passo=PASSO_CM):
    """
    Corrige rótulos usando a geometria.
    Retorna (dets_corrigidas, ajuste, relatorio).
    relatorio: lista de strings descrevendo cada correção feita.
    """
    relatorio = []
    if len(dets) < 2:
        return list(dets), None, relatorio

    # a reta é ajustada preferindo detecções confiantes
    base = [d for d in dets if d.conf >= 0.5]
    ajuste = ajustar_robusto(base if len(base) >= 2 else dets, tol_cm)
    if ajuste is None or len(ajuste.inliers) < 2:
        return list(dets), ajuste, relatorio

    px_cm = ajuste.px_por_cm
    tol_px = tol_cm * px_cm
    saida = []
    for d in dets:
        esperado_bruto = ajuste.valor_em(d.cy)
        esperado = _arredondar_passo(esperado_bruto, passo)
        desvio_px = abs(d.cy - ajuste.y_de(d.valor))
        if desvio_px <= tol_px or d.valor == esperado:
            saida.append(d)                       # coerente
            continue
        # incoerente: a posição diz outro valor
        residuo_cm = abs(esperado_bruto - esperado)
        if not (0 <= esperado <= 100) or residuo_cm > tol_cm:
            relatorio.append(f"descartado '{d.classe}' (conf {d.conf:.2f}): "
                             f"posição não corresponde a nenhuma marcação")
            continue
        if d.conf >= CONF_INTOCAVEL and desvio_px <= 2 * tol_px:
            saida.append(d)                       # confiante e quase coerente
            continue
        relatorio.append(f"corrigido '{d.classe}' -> '{esperado}' "
                         f"(conf {d.conf:.2f}; posição indica {esperado_bruto:.1f} cm)")
        saida.append(replace(d, classe=str(esperado)))

    # duplicatas: mantém a de maior confiança
    por_valor = {}
    for d in saida:
        atual = por_valor.get(d.valor)
        if atual is None or d.conf > atual.conf:
            if atual is not None:
                relatorio.append(f"duplicata de '{d.classe}' removida "
                                 f"(mantida a de conf {max(d.conf, atual.conf):.2f})")
            por_valor[d.valor] = d
        else:
            relatorio.append(f"duplicata de '{d.classe}' removida "
                             f"(conf {d.conf:.2f} < {atual.conf:.2f})")
    saida = sorted(por_valor.values(), key=lambda d: d.valor)

    # reajusta a reta com os valores corrigidos
    ajuste_final = ajustar_robusto(saida, tol_cm) or ajuste
    return saida, ajuste_final, relatorio


def detectar_lacunas(dets, ajuste, passo=PASSO_CM, tol_cm=TOL_CM):
    """
    Encontra números que deveriam estar visíveis entre o menor e o maior
    detectado, mas não foram detectados (miss do detector).
    Retorna lista de valores faltantes.
    """
    if not dets or ajuste is None:
        return []
    valores = sorted(d.valor for d in dets)
    faltantes = [v for v in range(valores[0], valores[-1] + 1, passo)
                 if v not in valores]
    return faltantes


def nivel_por_geometria(dets, ajuste, lacunas, passo=PASSO_CM):
    """
    Estima o nível considerando que pode haver um número submerso não detectado
    logo abaixo do menor visível.

    Retorna (nivel_cm, observacao).
    """
    if not dets or ajuste is None:
        return None, "sem ajuste geométrico"
    menor = min(dets, key=lambda d: d.valor)
    px_cm = ajuste.px_por_cm

    # a borda inferior do menor número visível aproxima a linha d'água
    nivel = menor.valor - (menor.y2 - menor.cy) / px_cm

    obs = f"px/cm={px_cm:.2f} r2={ajuste.r2:.3f} inliers={len(ajuste.inliers)}"
    if lacunas:
        obs += f" | lacunas detectadas: {lacunas}"
    return float(np.clip(nivel, 0, 100)), obs