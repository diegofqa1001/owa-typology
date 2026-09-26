"""Pruebas de la cadena del Capítulo 3 de la tesis (tesis_cap3/cadena_cap3.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cadena_cap3 as cc  # noqa: E402
from owa_typology import LABELS, rim_weights, orness  # noqa: E402

COG = {k: cc.cog_trapecio(*v) for k, v in LABELS.items()}


def _direccional(vec):
    t = np.array([COG[l] for l in vec])
    for i in cc.INVERTIDAS:
        t[i] = 1 - t[i]
    return t


def test_cog_forma_cerrada():
    assert COG["VL"] == pytest.approx(0.0975 / 1.05)
    assert COG["M"] == pytest.approx(0.5)
    assert COG["VL"] + COG["VH"] == pytest.approx(1.0)


def test_regla_lexicografica_reproduce_orden_de_la_tesis():
    orden = sorted(cc.PERFILES, key=lambda p: (
        round(_direccional(cc.PERFILES[p][1])[list(cc.NUCLEO)].mean(), 12),
        round(_direccional(cc.PERFILES[p][1])[list(cc.MODULADORAS)].mean(), 12)))
    assert orden == cc.ORDEN_TESIS


def test_media_uniforme_invierte_el_trio_intermedio():
    orden = sorted(cc.PERFILES, key=lambda p: _direccional(cc.PERFILES[p][1]).mean())
    assert orden[3:6] == ["P5", "P4", "P6"]


def test_centroides_de_diseno_respetan_el_orden():
    c = [cc.PERFILES[p][2] for p in cc.ORDEN_TESIS]
    assert all(b > a for a, b in zip(c, c[1:]))


def test_alfa_exacto_alcanza_el_centroide():
    for p in cc.ORDEN_TESIS:
        c = cc.PERFILES[p][2]
        assert orness(rim_weights(cc.alfa_exacto(c))) == pytest.approx(c, abs=1e-10)


def test_congelados_fuera_de_tolerancia_solo_innovador_y_visionario():
    fuera = [p for p in cc.ORDEN_TESIS
             if abs(orness(rim_weights(cc.ALFA_CONGELADO[p])) - cc.PERFILES[p][2]) >= cc.EPS]
    assert fuera == ["P7", "P8"]
