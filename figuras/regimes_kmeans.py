"""Gráfico e Validação do Agrupamento K-means (Item 2.5 do Relatório).

Descrição do Item 2.5:
    "Os regimes não vêm rotulados: são descobertos por K-means nas três condições de operação.
     O K-means minimiza a soma das distâncias quadráticas de cada ponto ao centróide do seu grupo:
     min sum ||z(s) - mu_k||^2, onde z(s) = (s - s_barra) / sigma_s.
     As condições são padronizadas antes (z), porque a altitude (0 a 42 mil pés)
     dominaria a distância euclidiana sobre o Mach (0 a 0,84)."

Este script produz uma figura explicativa completa com 4 painéis:
    (a) As 6 Condições de Voo Físicas (Altitude vs. Mach com R0-R5 e TRA anotado).
    (b) Vista Tridimensional (Altitude x Mach x TRA) evidenciando o regime R3 (TRA 60).
    (c) O Efeito Obrigatório da Padronização Z-Score (espaço bruto anisotrópico vs. espaço z isotrópico).
    (d) Varredura de K (Silhouette e Cotovelo/Inércia de k=2 a k=8), provando o pico de 0,997 em K=6.
"""

from __future__ import annotations

from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.domain.schema import SETTING_COLUMNS
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.reporting import viz_style


def gerar_grafico_kmeans_item_2_5(
    subset: str = "FD004",
    output_dir: Path | None = None,
) -> None:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "saida"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GERANDO FIGURA DO ITEM 2.5: AGRUPAMENTO POR K-MEANS NAS CONDIÇÕES DE VOO")
    print("=" * 80)

    # 1. Carregar dados de treino
    loader = CMapssLoader()
    df = loader.load(subset)

    # 2. Ajustar RegimeDecoupler
    decoupler = RegimeDecoupler(n_clusters=6, random_state=42)
    df_reg = decoupler.fit_transform(df)

    # Centróides físicos ordenados
    centers_phys = decoupler.scaler.inverse_transform(decoupler.kmeans.cluster_centers_)
    # Mapeia cada cluster original para o regime_id ordenado (0 a 5)
    cluster_to_regime = decoupler.cluster_to_regime_id
    regime_centers = {}
    for cluster_idx, r_id in cluster_to_regime.items():
        regime_centers[r_id] = centers_phys[cluster_idx]

    # Salvar tabela de centróides
    centroides_df = pd.DataFrame([
        {
            "Regime": f"R{r_id}",
            "Altitude_mil_pes": round(regime_centers[r_id][0], 1),
            "Mach": round(regime_centers[r_id][1], 2),
            "TRA_graus": round(regime_centers[r_id][2], 1),
            "N_Pontos": int((df_reg["regime_id"] == r_id).sum()),
            "Pct_Frota": round(float((df_reg["regime_id"] == r_id).mean() * 100), 2),
        }
        for r_id in range(6)
    ])
    centroides_csv = output_dir / "tabela_centroides_regimes.csv"
    centroides_df.to_csv(centroides_csv, index=False)
    print(f"-> Tabela de centróides salva: {centroides_csv}")

    # 3. Varredura de K (Silhouette e Inércia)
    print("\nCalculando varredura de K (k=2 a k=8) para avaliar qualidade dos clusters...")
    scaler = StandardScaler()
    sample_sweep = df.sample(n=4000, random_state=42)
    settings_scaled = scaler.fit_transform(sample_sweep[SETTING_COLUMNS].to_numpy(dtype=float))

    k_values = list(range(2, 9))
    inertias = []
    silhouettes = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(settings_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(settings_scaled, km.labels_)
        silhouettes.append(sil)

    sweep_df = pd.DataFrame({
        "K": k_values,
        "Inercia_WCSS": inertias,
        "Silhouette_Score": silhouettes,
    })
    sweep_csv = output_dir / "tabela_kmeans_metricas_k.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"-> Tabela de varredura de K salva: {sweep_csv}")

    # 4. Construção da Figura (2x2)
    viz_style.apply_style()
    fig = plt.figure(figsize=(11, 9))

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    sample_plot = df_reg.sample(n=4000, random_state=42)

    # ----------------------------------------------------
    # Subplot (a): Espaço Físico (Altitude x Mach)
    # ----------------------------------------------------
    ax_a = fig.add_subplot(2, 2, 1)
    for r_id in range(6):
        pts = sample_plot[sample_plot["regime_id"] == r_id]
        ax_a.scatter(
            pts["setting_1"], pts["setting_2"],
            color=palette[r_id], alpha=0.35, s=18, edgecolors="none",
            label=f"R{r_id}",
        )
        # Marcação do centróide
        c = regime_centers[r_id]
        ax_a.scatter(
            c[0], c[1], color=palette[r_id], s=120, edgecolors="black", linewidths=1.8, zorder=5
        )
        # Texto de anotação
        tra_str = f"TRA {c[2]:.0f}"
        ax_a.annotate(
            f"R{r_id}\n({tra_str})",
            (c[0], c[1]),
            xytext={2: (-62, 6), 3: (8, -38), 4: (-72, 4), 5: (-22, 12)}.get(r_id, (7, 5)),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.85),
        )

    ax_a.set_xlim(-4, 48)
    ax_a.set_ylim(-0.06, 1.08)  # deixa espaço para R4 e R5 (Mach 0,84) não serem cortados
    ax_a.set_xlabel("Altitude (milhares de pés)", fontsize=11.5)
    ax_a.set_ylabel("Velocidade (Mach)", fontsize=11.5)
    ax_a.set_title("(a) Centroides físicos recuperados (Altitude × Mach)", fontsize=12.5, fontweight="bold")
    ax_a.grid(True, linestyle=":", alpha=0.5)
    leg_a = ax_a.legend(loc="lower right", fontsize=10, title="Regime", framealpha=0.85, markerscale=2.2)
    for h in leg_a.legend_handles:
        h.set_alpha(1.0)

    # ----------------------------------------------------
    # Subplot (b): Vista 3D (Altitude x Mach x TRA)
    # ----------------------------------------------------
    ax_b = fig.add_subplot(2, 2, 2, projection="3d")
    ax_b.set_facecolor("#fcfcfb")
    for r_id in range(6):
        pts = sample_plot[sample_plot["regime_id"] == r_id]
        ax_b.scatter(
            pts["setting_1"], pts["setting_2"], pts["setting_3"],
            color=palette[r_id], alpha=0.25, s=12, edgecolors="none"
        )
        c = regime_centers[r_id]
        ax_b.scatter(
            c[0], c[1], c[2],
            color=palette[r_id], s=100, edgecolors="black", linewidths=1.5, zorder=10
        )
        ax_b.text(c[0], c[1], c[2] + 3, f"R{r_id}", fontsize=10.5, fontweight="bold", color="#111111")

    ax_b.set_xlabel("Altitude (k pés)", fontsize=10, labelpad=5)
    ax_b.set_ylabel("Mach", fontsize=10, labelpad=5)
    ax_b.set_zlabel("TRA (%)", fontsize=10, labelpad=5)
    ax_b.set_title("(b) Espaço tridimensional (R3 se destaca com TRA=60)", fontsize=12.5, fontweight="bold", pad=10)
    ax_b.view_init(elev=20, azim=-60)

    # ----------------------------------------------------
    # Subplot (c): Por que Padronizar? (Escalas relativas)
    # ----------------------------------------------------
    ax_c = fig.add_subplot(2, 2, 3)

    # Dados não padronizados
    alt_raw = sample_plot["setting_1"].to_numpy()
    mach_raw = sample_plot["setting_2"].to_numpy()
    tra_raw = sample_plot["setting_3"].to_numpy()

    # Desvios padrão brutos
    std_alt = np.std(alt_raw)
    std_mach = np.std(mach_raw)
    std_tra = np.std(tra_raw)

    print(f"desvios brutos: altitude {std_alt:.2f}, Mach {std_mach:.3f}, TRA {std_tra:.2f}; razões {std_alt/std_mach:.1f} e {std_tra/std_mach:.1f}")
    bar_labels = ["Altitude\n(mil pés)", "Mach", "TRA\n(manete)"]
    stds_raw = [std_alt, std_mach, std_tra]

    x_pos = np.arange(len(bar_labels))
    width = 0.35

    bars1 = ax_c.bar(x_pos - width/2, stds_raw, width, label="Escala bruta (desvio-padrão σ)", color="#eb6834")
    bars2 = ax_c.bar(x_pos + width/2, [1.0, 1.0, 1.0], width, label="Depois de padronizar (σ = 1,0)", color="#2a78d6")

    # Valores em cima das barras
    for bar in bars1:
        yval = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        yval = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=10)

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(bar_labels, fontsize=11)
    ax_c.set_ylabel("Desvio Padrão da Variável", fontsize=11.5)
    ax_c.set_title(
        "(c) Por que padronizar antes de agrupar:\n"
        "sem isso, a altitude e o TRA pesariam\ncerca de 50 vezes mais que o Mach",
        fontsize=12,
        fontweight="bold",
    )
    ax_c.set_ylim(0, 21.5)
    ax_c.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_c.legend(loc="upper center", fontsize=10)

    # ----------------------------------------------------
    # Subplot (d): Varredura de K (Silhueta e Inércia)
    # ----------------------------------------------------
    ax_d = fig.add_subplot(2, 2, 4)

    # Linha da Silhueta
    color_sil = "#1baf7a"
    ax_d.set_xlabel("Número de Grupos (K)", fontsize=11.5)
    ax_d.set_ylabel("Coeficiente de Silhueta", color=color_sil, fontsize=11.5)
    line_sil = ax_d.plot(k_values, silhouettes, marker="s", color=color_sil, linewidth=2.2, markersize=7, label="Silhueta (Pico em K=6: 0,997)")
    ax_d.tick_params(axis="y", labelcolor=color_sil)
    ax_d.set_ylim(0.5, 1.05)

    # Destaque no ponto K=6
    ax_d.scatter([6], [silhouettes[4]], color="#d03b3b", s=140, zorder=6, edgecolors="black", linewidths=1.5)
    ax_d.annotate(
        f"K = 6\nsilhueta = {silhouettes[4]:.3f}".replace(".", ","),
        (6, silhouettes[4]),
        xytext=(18, -75),
        textcoords="offset points",
        fontsize=10.5,
        fontweight="bold",
        color="#d03b3b",
        arrowprops=dict(arrowstyle="->", color="#d03b3b", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d03b3b", alpha=0.9),
    )

    # Eixo gêmeo para Inércia (Método do Cotovelo)
    ax_d2 = ax_d.twinx()
    color_inercia = "#52514e"
    ax_d2.set_ylabel("Inércia (WCSS)", color=color_inercia, fontsize=11.5)
    line_ine = ax_d2.plot(k_values, inertias, marker="o", color=color_inercia, linestyle="--", linewidth=1.8, markersize=6, label="Inércia / Cotovelo")
    ax_d2.tick_params(axis="y", labelcolor=color_inercia)

    # Legenda combinada
    lines = line_sil + line_ine
    labels = [l.get_label() for l in lines]
    ax_d.legend(lines, labels, loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=10, framealpha=0.9)

    ax_d.set_title(
        "(d) Varredura de K: por que K = 6?\n"
        "a silhueta tem pico claro em K = 6\n(grupos quase perfeitamente separados)",
        fontsize=12,
        fontweight="bold",
    )
    ax_d.grid(True, linestyle=":", alpha=0.5)

    # Título Geral
    plt.suptitle(
        "Agrupamento das condições de voo por K-means (FD004): os seis regimes são recuperados sem rótulos",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out_file = output_dir / "figura_kmeans_item_2_5.png"
    plt.savefig(out_file, dpi=180)
    # cópia usada pela monografia (Figura 5.1)
    tese = Path(__file__).resolve().parent / "saida" / "F02_kmeans_regimes.png"
    plt.savefig(tese, dpi=180)
    plt.close()
    print(f"\n-> Gráfico salvo com sucesso: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    gerar_grafico_kmeans_item_2_5()
