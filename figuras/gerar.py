"""Gera as figuras da monografia a partir das tabelas e dos resultados deste repositório.

    python figuras/gerar.py --lista        # tabela Figura -> arquivo -> script
    python figuras/gerar.py 5.17           # gera só a Figura 5.17 (roda o script que a produz)
    python figuras/gerar.py --todas        # gera todas as que têm script

As imagens vão para figuras/saida/. As figuras estáticas (fotos e diagramas) não têm script aqui:
ficam no repositório do texto (ver MANIFESTO.md).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent

# número no PDF -> (arquivo gerado, script, argumentos)
FIGURAS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "1.1": ("F24_bruto_vs_residuo.png", "novas.py", ()),
    "4.3": ("F03_metodo_a_regimes.png", "metodo_a_regimes.py", ()),
    "4.5": ("F21a_injecao_serie_saudavel.png", "injecao_series.py", ()),
    "4.6": ("F21b_injecao_formas_de_falha.png", "injecao_series.py", ()),
    "5.1": ("F02_kmeans_regimes.png", "regimes_kmeans.py", ()),
    "5.2": ("F06_comparacao_modelos_ic_fd004.png", "consistencia.py", ()),
    "5.3": ("F15_dispersao_predito_real_xgboost_temporal.png", "revisao.py", ("--so", "f15")),
    "5.4": ("F05_carta_controle_residuo_fd004.png", "revisao.py", ("--so", "f05")),
    "5.5": ("F04_motor179_queda_ciclo242.png", "motor179_queda.py", ()),
    "5.6": ("F14_linha1.png", "revisao.py", ("--so", "f14")),
    "5.7": ("F14_linha2.png", "revisao.py", ("--so", "f14")),
    "5.8": ("F14_linha3.png", "revisao.py", ("--so", "f14")),
    "5.9": ("F14_linha4.png", "revisao.py", ("--so", "f14")),
    "5.10": ("F17a_envelope_motor_completo.png", "revisao.py", ("--so", "f17")),
    "5.11": ("F17b_envelope_desvios_isolados.png", "revisao.py", ("--so", "f17")),
    "5.12": ("F17c_envelope_persistencia_frota.png", "revisao.py", ("--so", "f17")),
    "5.13": ("F23_decomposicao_ganho.png", "consistencia.py", ()),
    "5.16": ("F11_curvas_operacionais.png", "deteccao.py", ()),
    "5.17": ("F12_deteccao_acumulada.png", "deteccao.py", ()),
    "5.18": ("F07_trajetorias_intervalo_conformal.png", "consistencia.py", ()),
    "5.19": ("F27_conformal_ajustado_cqr.png", "consistencia.py", ()),
    "5.20": ("F26_custo_por_limiar.png", "novas.py", ()),
    "B.8": ("F18_ganho_ml_vs_media_regime.png", "consistencia.py", ()),
    "B.11": ("F08_criticidade_tradeoff.png", "consistencia.py", ()),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("figura", nargs="?", help="número da figura no PDF, ex.: 5.17")
    ap.add_argument("--lista", action="store_true")
    ap.add_argument("--todas", action="store_true")
    a = ap.parse_args()
    if a.lista or not (a.figura or a.todas):
        for n, (arq, script, args) in FIGURAS.items():
            print(f"Figura {n:>5}  {arq:50s} figuras/{script} {' '.join(args)}")
        return 0
    if a.todas:
        # um script gera várias figuras: roda cada um uma vez, sem filtro (revisao.py sem --so faz todas as suas)
        jobs = [(sc, ()) for sc in dict.fromkeys(sc for _, sc, _ in FIGURAS.values())]
    elif a.figura in FIGURAS:
        jobs = [(FIGURAS[a.figura][1], FIGURAS[a.figura][2])]
    else:
        print(f"Figura {a.figura}: sem script (estática ou fora da lista).", file=sys.stderr)
        return 2
    (AQUI / "saida").mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src"), "MPLBACKEND": "Agg"}
    for script, args in jobs:
        print(f"> figuras/{script} {' '.join(args)}", flush=True)
        r = subprocess.run([sys.executable, str(AQUI / script), *args], cwd=RAIZ, env=env)
        if r.returncode:
            return r.returncode
    faltam = [arq for n, (arq, _, _) in FIGURAS.items() if (a.todas or n == a.figura) and not (AQUI / "saida" / arq).exists()]
    if faltam:
        print("NÃO GERADAS:", *faltam, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
