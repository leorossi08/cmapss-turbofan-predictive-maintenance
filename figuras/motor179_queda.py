"""Demonstração Gráfica da Queda Abrupta no Ciclo 242 do Motor 179 (FD004).

Este script gera uma figura científica de 4 quadrantes comprovando visualmente
e estatisticamente o que ocorreu no ciclo 242 do Motor 179:
    1. O motor inteiro tem uma descontinuidade no ciclo 242, mas com magnitudes muito diferentes. Em % do valor
       do sensor o núcleo muda de 0,6 a 2,4 % (Nc, T30, T50, P30, phi), enquanto Nf (-0,02 %) e T24 (-0,12 %)
       quase não mudam; o BPR muda 1,5 %. Em desvios-padrão da janela saudável, porém, Nf (-10 σ) e BPR (+8 σ)
       também saltam, porque a variância saudável deles é minúscula. "Fan intacto" NÃO é uma conclusão correta.
    2. O núcleo (compressor de alta HPC e câmara) sofreu rejuvenescimento abrupto (Nc, T30, T50 e phi despencam).
    3. Os resíduos voltam à faixa de calibração saudável de motor novo (ciclos 1-50).
    4. A partir do ciclo 242, o motor inicia um segundo ciclo completo de degradação até a falha no ciclo 435.
"""

from __future__ import annotations

from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.reporting import viz_style


def gerar_grafico_rejuvenescimento_179(output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "saida"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GERANDO FIGURA DE DEMONSTRAÇÃO: REJUVENESCIMENTO DO MOTOR 179 (CICLO 242)")
    print("=" * 80)

    # 1. Carregar dados de FD004 e aplicar RegimeDecoupler
    loader = CMapssLoader()
    df = loader.load("FD004")
    decoupler = RegimeDecoupler(n_clusters=6, random_state=42)
    df_reg = decoupler.fit_transform(df)

    m179 = df_reg[df_reg["unit_id"] == 179].sort_values("cycle").copy().reset_index(drop=True)

    # 2. Descontar o regime para obter resíduos puros
    monitored_sensors = ["T24", "T30", "T50", "P30", "Ps30", "Nf", "Nc", "phi", "BPR"]
    for s in monitored_sensors:
        base = m179.groupby("regime_id")[s].transform("mean")
        m179[s + "_res"] = m179[s] - base
        m179[s + "_res_ma5"] = m179[s + "_res"].rolling(5, min_periods=1).mean()

    # 3. Estatísticas Pré vs Pós ciclo 242
    pre = m179[(m179["cycle"] >= 210) & (m179["cycle"] <= 241)]
    post = m179[(m179["cycle"] >= 242) & (m179["cycle"] <= 270)]
    healthy = m179[m179["cycle"] <= 50]

    delta_dict = {}
    for s in monitored_sensors:
        val_pre = pre[s + "_res"].mean()
        val_post = post[s + "_res"].mean()
        base_s = m179.groupby("regime_id")[s].transform("mean")
        pct = 100.0 * m179[s + "_res"] / base_s
        delta_dict[s] = {
            "Pre_242": val_pre,
            "Post_242": val_post,
            "Delta": val_post - val_pre,
            "Healthy_Mean": healthy[s + "_res"].mean(),
            "Healthy_Std": healthy[s + "_res"].std(),
            # magnitude relativa (em % do valor do sensor) e em desvios-padrão da janela saudável
            "Delta_pct": pct[post.index].mean() - pct[pre.index].mean(),
            "Delta_sigma": (val_post - val_pre) / healthy[s + "_res"].std(),
        }
    delta_df = pd.DataFrame(delta_dict).T.reset_index().rename(columns={"index": "Sensor"})
    delta_csv = output_dir / "tabela_impacto_ciclo_242_motor179.csv"
    delta_df.to_csv(delta_csv, index=False)
    print(f"-> Tabela de impacto salva: {delta_csv}")

    # 4. Construção da Figura
    viz_style.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    cycles = m179["cycle"].to_numpy()

    # ----------------------------------------------------
    # Subplot (a): Sensores do Núcleo (HPC / Combustor / LPT)
    # ----------------------------------------------------
    ax_core = axes[0, 0]
    ax_core.plot(cycles, m179["Nc_res_ma5"], label="Nc: Rotação Núcleo (rpm)", color="#d03b3b", lw=2.2)
    ax_core.plot(cycles, m179["T50_res_ma5"] * 2.0, label="T50: Temp. Escape (x2 °R)", color="#eb6834", lw=2.0)
    ax_core.plot(cycles, m179["T30_res_ma5"] * 2.0, label="T30: Temp. Saída HPC (x2 °R)", color="#fab219", lw=1.8, linestyle="--")

    # Faixa saudável (ciclos 1 a 50)
    ax_core.axvspan(1, 50, color="#1baf7a", alpha=0.12, label="Janela Saudável Inicial (1-50)")

    # Marco do Ciclo 242
    ax_core.axvline(242, color="#7c3aed", lw=2.0, linestyle=":", zorder=5)
    ax_core.annotate(
        f"Ciclo 242: queda média de\n{abs(delta_dict['Nc']['Delta']):.0f} rpm (Nc) e {abs(delta_dict['T50']['Delta']):.0f} °R (T50),\nde volta ao nível do início\n(ciclos 242–270 vs. 210–241)",
        xy=(242, 35),
        xytext=(252, 24),
        arrowprops=dict(facecolor="#7c3aed", shrink=0.05, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#7c3aed",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#7c3aed", alpha=0.9),
    )

    # Anotações das 2 fases de degradação
    ax_core.text(145, 0.03, "1ª degradação contínua\n(ciclos 50 a 241)", fontsize=10, ha="center", va="bottom", color="#52514e", fontweight="semibold", transform=ax_core.get_xaxis_transform())
    ax_core.text(340, 0.03, "2ª degradação contínua\n(ciclos 242 a 435, fim de vida)", fontsize=10, ha="center", va="bottom", color="#52514e", fontweight="semibold", transform=ax_core.get_xaxis_transform())

    ax_core.set_title("(a) Eixo de alta pressão (núcleo): queda brusca no ciclo 242", fontsize=12, fontweight="bold")
    ax_core.set_xlabel("Ciclo Operacional do Motor 179", fontsize=10.5)
    ax_core.set_ylabel("resíduo sem o efeito do regime\n(Nc em rpm; T30 e T50 em °R × 2)", fontsize=10.5)
    ax_core.grid(True, linestyle=":", alpha=0.5)
    ax_core.set_ylim(-45, 58)
    ax_core.legend(loc="upper left", fontsize=9.5, framealpha=0.9)

    # ----------------------------------------------------
    # Subplot (b): Sensores de Baixa Pressão (Fan + LPC)
    # ----------------------------------------------------
    ax_fan = axes[0, 1]
    ax_fan.plot(cycles, m179["Nf_res_ma5"], label="Nf: Rotação do Fan (rpm)", color="#2a78d6", lw=2.2)
    ax_fan.plot(cycles, m179["T24_res_ma5"], label="T24: Temp. Saída LPC (°R)", color="#1baf7a", lw=2.0)

    ax_fan.axvspan(1, 50, color="#1baf7a", alpha=0.12, label="Janela Saudável Inicial")
    ax_fan.axvline(242, color="#7c3aed", lw=2.0, linestyle=":", zorder=5)

    ax_fan.annotate(
        f"Ciclo 242: variação pequena em valor relativo\nΔ Nf = {delta_dict['Nf']['Delta']:.2f} rpm ({delta_dict['Nf']['Delta_pct']:.2f} %; {delta_dict['Nf']['Delta_sigma']:.0f} σ)\nΔ T24 = {delta_dict['T24']['Delta']:.2f} °R ({delta_dict['T24']['Delta_pct']:.2f} %; {delta_dict['T24']['Delta_sigma']:.0f} σ)\nA descontinuidade existe em σ".replace(".", ","),
        xy=(242, 0.0),
        xytext=(40, 2.6),
        ha="left",
        arrowprops=dict(facecolor="#2a78d6", shrink=0.05, width=1.5, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
        color="#2a78d6",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2a78d6", alpha=0.9),
    )
    ax_fan.set_xlim(0, max(cycles) + 25)

    ax_fan.set_title("(b) Eixo de baixa (fan e LPC): pequena em %, visível em σ", fontsize=12, fontweight="bold")
    ax_fan.set_xlabel("Ciclo Operacional do Motor 179", fontsize=10.5)
    ax_fan.set_ylabel("resíduo sem o efeito do regime\n(Nf em rpm; T24 em °R)", fontsize=10.5)
    ax_fan.set_ylim(-4, 6)
    ax_fan.grid(True, linestyle=":", alpha=0.5)
    ax_fan.legend(loc="lower right", fontsize=9.5, framealpha=0.9)

    # ----------------------------------------------------
    # Subplot (c): Zoom no Regime Fixo R4 (35k pés, Mach 0.84)
    # ----------------------------------------------------
    ax_zoom = axes[1, 0]
    r4_data = m179[(m179["regime_id"] == 4) & (m179["cycle"] >= 180) & (m179["cycle"] <= 300)].sort_values("cycle")

    ax_zoom.plot(r4_data["cycle"], r4_data["T50"], marker="o", color="#eb6834", lw=1.8, markersize=5, label="T50 Bruto no Regime 4 (°R)")
    ax_zoom.axvline(242, color="#7c3aed", lw=2.0, linestyle=":", zorder=5)

    # Destacar os dois pontos imediatos
    p_before = r4_data[r4_data["cycle"] == 241].iloc[0]
    p_after = r4_data[r4_data["cycle"] == 246].iloc[0]
    ax_zoom.scatter([p_before["cycle"]], [p_before["T50"]], color="#d03b3b", s=100, zorder=6, edgecolors="black")
    ax_zoom.scatter([p_after["cycle"]], [p_after["T50"]], color="#1baf7a", s=100, zorder=6, edgecolors="black")

    ax_zoom.annotate(
        f"Ciclo 241: {p_before['T50']:.2f} °R".replace(".", ","),
        xy=(p_before["cycle"], p_before["T50"]),
        xytext=(194, 1150.5),
        textcoords="data",
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#d03b3b"),
    )
    ax_zoom.annotate(
        f"Ciclo 246: {p_after['T50']:.2f} °R\n(queda de {abs(p_after['T50'] - p_before['T50']):.2f} °R entre\nestes dois pontos; o painel (a)\nusa médias de janelas de ciclos)".replace(".", ","),
        xy=(p_after["cycle"], p_after["T50"]),
        xytext=(249, 1141),
        textcoords="data",
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1baf7a"),
    )

    ax_zoom.set_title("(c) Só o regime 4: a queda aparece no mesmo regime", fontsize=12, fontweight="bold")
    ax_zoom.set_xlabel("Ciclo Operacional (Faixa 180 a 300)", fontsize=10.5)
    ax_zoom.set_ylabel("T50 Bruto (°R)", fontsize=10.5)
    ax_zoom.grid(True, linestyle=":", alpha=0.5)
    ax_zoom.legend(loc="lower left", fontsize=9.5)

    # ----------------------------------------------------
    # Subplot (d): Variação relativa (% do valor do sensor), núcleo x baixa pressão
    # ----------------------------------------------------
    ax_bar = axes[1, 1]
    names_d = {"Nf": "Nf (Fan)", "T24": "T24 (LPC)", "BPR": "BPR (bypass)", "Ps30": "Ps30", "phi": "phi (Combustão)", "P30": "P30 (HPC)", "T30": "T30 (HPC)", "T50": "T50 (LPT)", "Nc": "Nc (Núcleo)"}
    keys = sorted(names_d, key=lambda k: abs(delta_dict[k]["Delta_pct"]))
    deltas = [delta_dict[k]["Delta_pct"] for k in keys]
    colors_bar = ["#2a78d6" if k in ["Nf", "T24"] else ("#898781" if k == "BPR" else "#d03b3b") for k in keys]

    y_pos = np.arange(len(keys))
    ax_bar.barh(y_pos, deltas, color=colors_bar, alpha=0.85, edgecolor="#333333", height=0.6)
    ax_bar.axvline(0, color="#111111", lw=1.0)
    for y, k, d in zip(y_pos, keys, deltas):
        offset = 0.05 if d >= 0 else -0.05
        ax_bar.text(d + offset, y, f"{d:+.2f} %  ({delta_dict[k]['Delta_sigma']:+.0f} σ)".replace(".", ","), va="center", ha="left" if d >= 0 else "right", fontsize=9.5, fontweight="bold", color="#111111")
    ax_bar.set_xlim(min(deltas) * 1.9, max(deltas) * 2.4 + 0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([names_d[k] for k in keys], fontsize=10.5)
    ax_bar.set_xlabel("variação do resíduo médio, em % do valor do sensor\n(ciclos 242–270 contra 210–241); entre parênteses,\na mesma variação em σ da janela saudável", fontsize=9.5)
    ax_bar.set_title("(d) Núcleo (vermelho) muda 0,6 a 2,4 %;\nNf e T24 (azul) muito menos; BPR (cinza) 1,5 %", fontsize=10.5, fontweight="bold")
    ax_bar.grid(True, linestyle=":", alpha=0.5, axis="x")

    # ----------------------------------------------------
    # Título Geral
    plt.suptitle(
        "Motor 179 (FD004): descontinuidade abrupta no ciclo 242 (compatível com restauração/manutenção)\n"
        "O núcleo muda de 0,6 a 2,4 % e volta ao patamar saudável; Nf e T24 mudam muito menos em %, mas também saltam em σ.",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out_file = output_dir / "figura_rejuvenescimento_motor_179.png"
    plt.savefig(out_file, dpi=180)
    # cópia usada pela monografia (Figura 5.5)
    plt.savefig(Path(__file__).resolve().parent / "saida" / "F04_motor179_queda_ciclo242.png", dpi=180)
    plt.close()
    print(f"\n-> Gráfico de rejuvenescimento salvo com sucesso: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    gerar_grafico_rejuvenescimento_179()
