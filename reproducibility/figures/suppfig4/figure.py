import errno
import os
import pickle
from logging import Logger
from pathlib import Path
import cospar as cs

import hydra
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from omegaconf import DictConfig

from pyrovelocity.config import print_config_tree
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import rainbowplot
from pyrovelocity.utils import get_pylogger
from pyrovelocity.data import load_larry

from scvelo.plotting.velocity_embedding_grid import default_arrow
from pyrovelocity.plot import align_trajectory_diff
from pyrovelocity.plot import get_clone_trajectory
from pyrovelocity.plot import plot_posterior_time, set_colorbar
from pyrovelocity.plot import plot_vector_field_uncertain, get_posterior_sample_angle_uncertainty


"""Loads trained figure S4 data and produces figure S4.
"""


def evaluate_time(adata_scvelo, adata_cytotrace, posterior_samples, gold, gold_select):
    all_metrics = np.zeros((2, 2))
    for i, gold_standard in enumerate([-gold[gold_select]]):
        for j, pred in enumerate(
            [
                adata_scvelo.obs.latent_time.values[gold_select],
                posterior_samples["cell_time"].mean(0).flatten()[gold_select],
            ]
        ):
            all_metrics[i, j] = spearmanr(gold_standard, pred)[0]
    for _, gold_standard in enumerate([1 - adata_cytotrace.obs.cytotrace]):
        for j, pred in enumerate(
            [
                adata_scvelo.obs.latent_time.values,
                posterior_samples["cell_time"].mean(0).flatten(),
            ]
        ):
            all_metrics[1, j] = spearmanr(gold_standard, pred)[0]
    all_metrics = pd.DataFrame(
        all_metrics,
        index=["Fate potency", "Cytotrace"],
        columns=["scVelo", "Pyro-Velocity"],
    )
    print(all_metrics)
    return all_metrics


def plots(conf: DictConfig, logger: Logger) -> None:
    ##################
    # load data
    ##################
    cs.logging.print_version()
    cs.settings.verbosity = 2
    cs.settings.data_path = "LARRY_data"  # A relative path to save data.
    cs.settings.figure_path = "LARRY_figure"  # A relative path to save figures.
    Path("LARRY_figure").mkdir(parents=True, exist_ok=True)
    Path("LARRY_data").mkdir(parents=True, exist_ok=True)
    cs.settings.set_figure_params(
        format="png", figsize=[4, 3.5], dpi=75, fontsize=14, pointsize=2
    )
    suppfig4_plot_name = conf.reports.suppfig4.suppfig4_pdf

    pyrovelocity_larry_data_path = (
        conf.model_training.larry_model2.pyrovelocity_data_path
    )
    trained_larry_data_path = conf.model_training.larry_model2.trained_data_path
    larry_cospar_data_path = conf.data_sets.larry_cospar.derived.rel_path
    larry_cytotrace_data_path = conf.data_sets.larry_cytotrace.derived.rel_path
    larry_dynamical_data_path = conf.data_sets.larry_dynamical.derived.rel_path

    # raw data
    adata = load_larry()
    adata_input_all = scv.read(trained_larry_data_path) #pyro-velocity
    adata_input_vel = scv.read(larry_dynamical_data_path) #scvelo
    adata_cospar = scv.read(larry_cospar_data_path) #cospar
    adata_cytotrace = scv.read(larry_cytotrace_data_path) #cytotrace

    cs.pl.fate_potency(
        adata_cospar,
        used_Tmap="transition_map",
        map_backward=True,
        method="norm-sum",
        color_bar=True,
        fate_count=True,
    )
    scv.tl.velocity_embedding(adata_input_vel, basis="emb")
    scv.pl.scatter(
        adata_input_all,
        basis="emb",
        fontsize=7,
        legend_loc="on data",
        legend_fontsize=7,
        color="state_info",
        show=False,
    )
        
    print_config_tree(conf.reports.figure3, logger, ())

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  reports: {conf.reports.figureS3.path}\n"
    )
    Path(conf.reports.suppfig4.path).mkdir(parents=True, exist_ok=True)

    posterior_samples = CompressedPickle.load(pyrovelocity_larry_data_path)
    v_map_all = posterior_samples["vector_field_posterior_samples"]
    embeds_radian_all = posterior_samples["embeds_angle"]
    fdri = posterior_samples["fdri"]
    pca_fdri = posterior_samples["pca_fdri"]
    embed_mean_all = posterior_samples["vector_field_posterior_mean"]
    
    if os.path.exists("global_gold_standard2.h5ad"):
        adata_input_all_clone = scv.read("global_gold_standard2.h5ad")
    else:
        adata_reduced_gene_for_clone_vec = adata[:, adata_input_vel.var_names].copy()
        print(adata_reduced_gene_for_clone_vec.shape)
        adata_input_all_clone = get_clone_trajectory(adata_reduced_gene_for_clone_vec)
        adata_input_all_clone.write("global_gold_standard2.h5ad")

    adata_cytotrace.obs.loc[:, "1-Cytotrace"] = 1 - adata_cytotrace.obs.cytotrace
    gold = adata_cospar[adata_input_all.obs_names.str.replace("-0", ""), :].obs.fate_potency
    gold_select = ~np.isnan(gold)

    metrics_all = evaluate_time(adata_input_vel, adata_cytotrace, posterior_samples, gold, gold_select)
    
    pyrovelocity_mono_data_path = conf.model_training.larry_mono_model2.pyrovelocity_data_path
    trained_mono_data_path = conf.model_training.larry_mono_model2.trained_data_path #pyro-velocity
    adata_mono_dynamical_data_path = conf.data_sets.larry_mono.derived.rel_path
    adata_uni_mono = scv.read(adata_mono_dynamical_data_path)
    posterior_samples = CompressedPickle.load(pyrovelocity_mono_data_path)

    gold_uni_mono = adata_cospar[
        adata_uni_mono.obs_names.str.replace("-0", ""), :
    ].obs.fate_potency
    gold_select_uni_mono = ~np.isnan(gold_uni_mono)
    metrics_mono =evaluate_time(adata_uni_mono, adata_uni_mono, posterior_samples, gold_uni_mono, gold_select_uni_mono)

    pyrovelocity_neu_data_path = conf.model_training.larry_neu_model2.pyrovelocity_data_path
    trained_neu_data_path = conf.model_training.larry_neu_model2.trained_data_path #pyro-velocity
    adata_neu_dynamical_data_path = conf.data_sets.larry_neu.derived.rel_path
    adata_uni_neu = scv.read(adata_neu_dynamical_data_path)
    posterior_samples = CompressedPickle.load(pyrovelocity_neu_data_path)
    gold_uni_neu = adata_cospar[
        adata_uni_neu.obs_names.str.replace("-0", ""), :
    ].obs.fate_potency
    gold_select_uni_neu = ~np.isnan(gold_uni_neu)
    metrics_neu = evaluate_time(adata_uni_neu, adata_uni_neu, posterior_samples, gold_uni_neu, gold_select_uni_neu)

    pyrovelocity_multilineage_data_path = conf.model_training.larry_multilineage_model2.pyrovelocity_data_path
    trained_multilineage_data_path = conf.model_training.larry_multilineage_model2.trained_data_path
    adata_multilineage_dynamical_data_path = scv.read(conf.data_sets.larry_multilineage.derived.rel_path)
    posterior_samples = CompressedPickle.load(pyrovelocity_multilineage_data_path)
    gold_multilineage = adata_cospar[
        adata_multilineage_dynamical_data_path.obs_names.str.replace(r"-\d", ""), :
    ].obs.fate_potency
    gold_multi_select = ~np.isnan(gold_multilineage)
    metrics_multi = evaluate_time(adata_multilineage_dynamical_data_path, adata_multilineage_dynamical_data_path, 
                  posterior_samples, gold_multilineage, gold_multi_select)

    fig = plt.figure(figsize=(9.6, 4))
    ax = fig.subplots(2, 4)
    scv.pl.scatter(
        adata_cytotrace,
        basis="emb",
        fontsize=7,
        color="1-Cytotrace",
        cmap="inferno_r",
        show=False,
        ax=ax[0][0],
        s=1,
    )
    ax[0][0].set_title(
        "Cytotrace\ncorrelation with fate potency: %.2f"
        % spearmanr(1 - adata_cytotrace.obs.cytotrace[gold_select], -gold[gold_select])[0],
        fontsize=7,
    )

    for index, fdr in enumerate([fdri, pca_fdri]):
        adata_input_all.obs.loc[:, "vector_field_rayleigh_test"] = fdr
        basis = "emb"
        im = ax[0][index+1].scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            s=3,
            alpha=0.9,
            c=adata_input_all.obs["vector_field_rayleigh_test"],
            cmap="inferno_r",
            linewidth=0,
        )
        #ax[0][index+1].text(
        #    -0.1,
        #    1.15,
        #    "b",
        #    transform=ax[0][1].transAxes,
        #    fontsize=7,
        #    fontweight="bold",
        #    va="top",
        #    ha="right",
        #)
        set_colorbar(im, ax[0][index+1], labelsize=5, fig=fig, position="right")
        ax[0][index+1].axis('off')

    ax[0][1].set_title('UMAP angle Rayleigh test', fontsize=7)
    ax[0][2].set_title('PCA angle Rayleigh test', fontsize=7)
    ax[0][3].axis('off')
    n = 0
    for title, metric in zip(['All', 'Monocyte', 'Neutrophil', 'Bifurcation'], [metrics_all, metrics_mono, metrics_neu, metrics_multi]):
        g = sns.heatmap(
            metric,
            annot=True,
            fmt=".3f",
            ax=ax[1][n],
            cbar=False,
            annot_kws={"fontsize": 7},
        )
        ax[1][n].set_xticklabels(
            ax[1][n].get_xmajorticklabels(), fontsize=7, rotation=0, ha="right"
        )
        ax[1][n].set_yticklabels(ax[1][0].get_ymajorticklabels(), fontsize=7)
        ax[1][n].set_title(title, fontsize=7)
        n += 1
    fig.subplots_adjust(
        hspace=0.25, wspace=0.5, left=0.01, right=0.92, top=0.93, bottom=0.1
    )
    fig.savefig(
        suppfig4_plot_name,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        edgecolor="none",
        dpi=300,
    )
    for ext in ["", ".png"]:
        fig.savefig(
            f"{suppfig4_plot_name}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


@hydra.main(version_base="1.2", config_path="..", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Plot results
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)

    if os.path.isfile(conf.reports.figureS3.tif_path) and os.path.isfile(
        conf.reports.figureS3.svg_path
    ):
        logger.info(
            f"\n\nFigure 2 outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figureS3.path}\n"
        )
    else:
        logger.info(f"\n\nPlotting figure S3\n\n")
        plots(conf, logger)


if __name__ == "__main__":
    main()
