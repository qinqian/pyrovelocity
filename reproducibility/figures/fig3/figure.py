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
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain, get_posterior_sample_angle_uncertainty


"""Loads trained figure S3 data and produces figure S3.

Inputs:
  data:
    "models/pancreas_model2/trained.h5ad"
    "models/pbmc68k_model2/trained.h5ad"
  models:
    "models/pancreas_model2/pyrovelocity.pkl"
    "models/pbmc68k_model2/pyrovelocity.pkl"

Outputs:
  figures:
    "reports/figS3/figS3_raw_gene_selection_model1.{tif,svg}"
"""


def plot_larry_subset(pyrovelocity_data_path, 
                      adata_pyrovelocity,
                      adata_scvelo,
                      adata_cospar, ax, fig,
                      state_color_dict, ylabel="Unipotent Monocyte lineage",
                      dotsize = 3, scale=0.35, arrow = 3.5):
    posterior_samples = CompressedPickle.load(pyrovelocity_data_path)
    v_map = posterior_samples["vector_field_posterior_samples"]
    embeds_radian = posterior_samples["embeds_angle"]
    fdri = posterior_samples["fdri"]
    embed_mean = posterior_samples["vector_field_posterior_mean"]

    adata_scvelo = scv.read(adata_scvelo)
    adata_pyrovelocity = scv.read(adata_pyrovelocity)
    adata_input_clone = get_clone_trajectory(adata_scvelo)
    adata_input_clone.obsm["clone_vector_emb"][
        np.isnan(adata_input_clone.obsm["clone_vector_emb"])
    ] = 0
    cutoff = 5
    density = 0.35
    diff = align_trajectory_diff(
        [adata_input_clone, adata_scvelo, adata_scvelo],
        [
            adata_input_clone.obsm["clone_vector_emb"],
            adata_scvelo.obsm["velocity_emb"],
            embed_mean,
        ],
        embed="emb",
        density=density,
    )
    scvelo_cos = pd.DataFrame(diff).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[4:6]), axis=1
    )
    pyro_cos = pd.DataFrame(diff).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[6:8]), axis=1
    )
    scvelo_cos_mean = scvelo_cos.mean()
    pyro_cos_mean = pyro_cos.mean()
    print(scvelo_cos_mean, pyro_cos_mean)

    res = pd.DataFrame(
        {
            "X": adata_pyrovelocity.obsm["X_emb"][:, 0],
            "Y": adata_pyrovelocity.obsm["X_emb"][:, 1],
            "celltype": adata_pyrovelocity.obs.state_info,
        }
    )
    sns.scatterplot(
        data=res,
        x="X",
        y="Y",
        hue="celltype",
        palette=state_color_dict,
        ax=ax[0],
        s=dotsize,
        alpha=0.90,
        linewidth=0,
        legend=False,
    )
    ax[0].set_title("Cell types", fontsize=7)
    ax[0].set_ylabel(ylabel, fontsize=7)
    scv.pl.velocity_embedding_grid(
        adata_input_clone,
        scale=scale,
        autoscale=True,
        show=False,
        s=dotsize,
        density=density,
        arrow_size=arrow,
        linewidth=1,
        vkey="clone_vector",
        basis="emb",
        ax=ax[1],
        title="Clonal progression",
        color="gray",
        arrow_color="black",
        fontsize=7,
    )
    # scvelo
    scv.pl.velocity_embedding_grid(
        adata_scvelo,
        show=False,
        s=dotsize,
        density=density,
        scale=scale,
        autoscale=True,
        arrow_size=arrow,
        linewidth=1,
        basis="emb",
        ax=ax[2],
        title="Scvelo",
        fontsize=7,
        color="gray",
        arrow_color="black",
    )
    ax[2].set_title("scVelo cosine similarity: %.2f" % scvelo_cos_mean, fontsize=7)
    cell_time_mean = posterior_samples['cell_time'].mean(0).flatten()
    cell_time_std = posterior_samples['cell_time'].std(0).flatten()
    cell_time_cov = cell_time_std / cell_time_mean
    print(cell_time_cov)

    plot_vector_field_uncertain(
        adata_pyrovelocity,
        embed_mean,
        cell_time_cov,
        ax=ax[3],
        cbar=True,
        fig=fig,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        only_grid=False,
        autoscale=True,
        uncertain_measure="shared time",
        cmap='winter'
    )
    ax[3].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_cos_mean, fontsize=7
    )
    cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
    cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
    cell_magnitudes_std = cell_magnitudes.std(axis=-2)
    cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
    plot_vector_field_uncertain(
        adata_pyrovelocity,
        embed_mean,
        cell_magnitudes_cov,
        ax=ax[4],
        cbar=True,
        fig=fig,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        only_grid=False,
        autoscale=True,
        uncertain_measure="base magnitude",
        cmap='summer'
    )
    ax[4].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_cos_mean, fontsize=7
    )

    pca_angles = posterior_samples["pca_embeds_angle"]
    pca_cell_angles = pca_angles / np.pi * 180 # degree
    pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)
    print(pca_angles_std)

    plot_vector_field_uncertain(
        adata_pyrovelocity,
        embed_mean,
        pca_angles_std,
        ax=ax[5],
        cbar=True,
        fig=fig,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        only_grid=False,
        autoscale=True,
        uncertain_measure="base magnitude",
        cmap='inferno', cmax=360
    )
    ax[5].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_cos_mean, fontsize=7
    )

    scv.pl.scatter(
        adata_cospar[adata_pyrovelocity.obs_names.str.replace(r"-\d", ""), :],
        basis="emb",
        fontsize=7,
        color="fate_potency",
        cmap="inferno_r",
        show=False,
        ax=ax[6],
        s=dotsize,
    )
    ax[6].set_title("Clonal fate potency", fontsize=7)
    gold = adata_cospar[
        adata_pyrovelocity.obs_names.str.replace(r"-\d", ""), :
    ].obs.fate_potency
    select = ~np.isnan(gold)
    scv.pl.scatter(
        adata_scvelo,
        c="latent_time",
        basis="emb",
        s=dotsize,
        cmap="inferno",
        ax=ax[7],
        show=False,
        fontsize=7,
    )
    ax[7].set_title(
        "Scvelo latent time\ncorrelation: %.2f"
        % spearmanr(-gold[select], adata_scvelo.obs.latent_time.values[select])[0],
        fontsize=7
    )
    plot_posterior_time(
        posterior_samples,
        adata_pyrovelocity,
        ax=ax[8],
        basis="emb",
        fig=fig,
        addition=False,
        position="right",
    )
    ax[8].set_title(
        "Pyro-Velocity shared time\ncorrelation: %.2f"
        % spearmanr(
            -gold[select], posterior_samples["cell_time"].mean(0).flatten()[select]
        )[0],
        fontsize=7,
    )


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
    figure3_plot_name = conf.reports.figure3.figure3_pdf

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
    Path(conf.reports.figure3.path).mkdir(parents=True, exist_ok=True)

    posterior_samples = CompressedPickle.load(pyrovelocity_larry_data_path)
    v_map_all = posterior_samples["vector_field_posterior_samples"]
    embeds_radian_all = posterior_samples["embeds_angle"]
    fdri = posterior_samples["fdri"]
    embed_mean_all = posterior_samples["vector_field_posterior_mean"]
    
    if os.path.exists("global_gold_standard2.h5ad"):
        adata_input_all_clone = scv.read("global_gold_standard2.h5ad")
    else:
        adata_reduced_gene_for_clone_vec = adata[:, adata_input_vel.var_names].copy()
        print(adata_reduced_gene_for_clone_vec.shape)
        adata_input_all_clone = get_clone_trajectory(adata_reduced_gene_for_clone_vec)
        adata_input_all_clone.write("global_gold_standard2.h5ad")
    
    adata_input_all_clone.obsm["clone_vector_emb"][
        np.isnan(adata_input_all_clone.obsm["clone_vector_emb"])
    ] = 0
    
    cutoff = 10
    density = 0.35
    diff_all = align_trajectory_diff(
        [adata_input_all_clone, adata_input_vel, adata_input_all],
        [
            adata_input_all_clone.obsm["clone_vector_emb"],
            adata_input_vel.obsm["velocity_emb"],
            embed_mean_all,
        ],
        embed="emb",
        density=density,
    )
    scvelo_all_cos = pd.DataFrame(diff_all).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[4:6]), axis=1
    )
    pyro_all_cos = pd.DataFrame(diff_all).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[6:8]), axis=1
    )
    scvelo_all_cos_mean = scvelo_all_cos.mean()
    pyro_all_cos_mean = pyro_all_cos.mean()

    color_dict = dict(
        zip(
            adata_input_all.obs.state_info.cat.categories,
            adata_input_all.uns["state_info_colors"],
        )
    )

    dotsize = 3
    scale = 0.35
    arrow = 3.5
    fig, ax = plt.subplots(4, 9)
    fig.set_size_inches(17, 11)
    fig.subplots_adjust(
        hspace=0.4, wspace=0.2, left=0.01, right=0.99, top=0.95, bottom=0.3
    )
    ax2 = ax[3]
    pyrovelocity_mono_data_path = conf.model_training.larry_mono_model2.pyrovelocity_data_path
    trained_mono_data_path = conf.model_training.larry_mono_model2.trained_data_path #pyro-velocity
    adata_mono_dynamical_data_path = conf.data_sets.larry_mono.derived.rel_path

    pyrovelocity_neu_data_path = conf.model_training.larry_neu_model2.pyrovelocity_data_path
    trained_neu_data_path = conf.model_training.larry_neu_model2.trained_data_path
    adata_neu_dynamical_data_path = conf.data_sets.larry_neu.derived.rel_path

    pyrovelocity_multilineage_data_path = conf.model_training.larry_multilineage_model2.pyrovelocity_data_path
    trained_multilineage_data_path = conf.model_training.larry_multilineage_model2.trained_data_path
    adata_multilineage_dynamical_data_path = conf.data_sets.larry_multilineage.derived.rel_path

    res = pd.DataFrame(
        {
            "X": adata_input_all.obsm["X_emb"][:, 0],
            "Y": adata_input_all.obsm["X_emb"][:, 1],
            "celltype": adata_input_all.obs.state_info,
        }
    )
    sns.scatterplot(
        data=res,
        x="X",
        y="Y",
        hue="celltype",
        palette=color_dict,
        s=dotsize,
        alpha=0.9,
        linewidth=0,
        legend="brief",
        ax=ax2[0],
    )
    ax2[0].set_title("Cell types", fontsize=7)
    scv.pl.velocity_embedding_grid(
        adata_input_all_clone,
        # scale=0.25,
        scale=scale,
        autoscale=True,
        show=False,
        s=dotsize,
        density=density,
        arrow_size=arrow,
        linewidth=1,
        vkey="clone_vector",
        basis="emb",
        ax=ax2[1],
        title="Clonal progression",
        color="gray",
        arrow_color="black",
        fontsize=7,
    )
    
    scv.pl.velocity_embedding_grid(
        # adata_input_all,
        adata_input_vel,
        show=False,
        s=dotsize,
        density=density,
        # scale=None,
        scale=scale,
        arrow_size=arrow,
        linewidth=1,
        basis="emb",
        ax=ax2[2],
        title="Scvelo",
        fontsize=7,
        color="gray",
        arrow_color="black",
        autoscale=True,
    )
    ax2[2].set_title("scVelo cosine similarity: %.2f" % scvelo_all_cos_mean, fontsize=7)

    cell_time_mean = posterior_samples['cell_time'].mean(0).flatten()
    cell_time_std = posterior_samples['cell_time'].std(0).flatten()
    cell_time_cov = cell_time_std / cell_time_mean

    plot_vector_field_uncertain(
        adata_input_all,
        embed_mean_all,
        cell_time_cov,
        ax=ax2[3],
        cbar=True,
        fig=fig,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        only_grid=False,
        autoscale=True,
        uncertain_measure="shared time",
        cmap='winter'
    )
    ax2[3].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_all_cos_mean, fontsize=7
    )

    cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
    cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
    cell_magnitudes_std = cell_magnitudes.std(axis=-2)
    cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean

    plot_vector_field_uncertain(
        adata_input_all,
        embed_mean_all,
        cell_magnitudes_cov,
        ax=ax2[4],
        cbar=True,
        fig=fig,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        only_grid=False,
        autoscale=True,
        uncertain_measure="base magnitude",
        cmap='summer'
    )
    ax2[4].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_all_cos_mean, fontsize=7
    )

    pca_angles = posterior_samples["pca_embeds_angle"]
    pca_cell_angles = pca_angles / np.pi * 180 # degree
    pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)

    plot_vector_field_uncertain(
        adata_input_all,
        embed_mean_all,
        pca_angles_std,
        ax=ax2[5],
        cbar=True,
        fig=fig,
        basis="emb",
        scale=scale,
        p_mass_min=1,
        density=density,
        arrow_size=arrow,
        only_grid=False,
        autoscale=True,
        uncertain_measure="base magnitude",
        cmap='inferno', cmax=360
    )
    ax2[5].set_title(
        "Pyro-Velocity cosine similarity: %.2f" % pyro_all_cos_mean, fontsize=7
    )

    #for a in ax2[3:6]:
    #    pos = a.get_position()
    #    a.text(
    #        pos.x0+pos.width*0.06, pos.y0-pos.height/13, "unspliced expression", size=7, va="center"
    #    )

    scv.pl.scatter(
        adata_cospar,
        basis="emb",
        fontsize=7,
        color="fate_potency",
        cmap="inferno_r",
        show=False,
        ax=ax2[6],
        s=dotsize,
    )
    ax2[6].set_title("Clonal fate potency", fontsize=7)
    gold = adata_cospar[adata_input_all.obs_names.str.replace("-0", ""), :].obs.fate_potency
    select = ~np.isnan(gold)
    scv.pl.scatter(
        adata_input_all,
        c="latent_time",
        basis="emb",
        s=dotsize,
        cmap="inferno",
        ax=ax2[7],
        show=False,
        fontsize=7,
    )
    ax2[7].set_title(
        "Scvelo latent time\ncorrelation: %.2f"
        % spearmanr(-gold[select], adata_input_all.obs.latent_time.values[select])[0],
        fontsize=7,
    )
    adata_input_all.obs.cytotrace = adata_cytotrace.obs.cytotrace
    plot_posterior_time(
        posterior_samples,
        adata_input_all,
        ax=ax2[8],
        basis="emb",
        fig=fig,
        addition=False,
        position="right",
    )
    ax2[8].set_title(
        "Pyro-Velocity shared time\ncorrelation: %.2f"
        % spearmanr(
            -gold[select], posterior_samples["cell_time"].mean(0).flatten()[select]
        )[0],
        fontsize=7,
    )

    print("mono")
    plot_larry_subset(pyrovelocity_mono_data_path, trained_mono_data_path, 
                      adata_mono_dynamical_data_path, adata_cospar, ax[0], fig, color_dict)
    print("neu")
    plot_larry_subset(pyrovelocity_neu_data_path, trained_neu_data_path, 
                      adata_neu_dynamical_data_path, adata_cospar, ax[1], fig, color_dict)
    plot_larry_subset(pyrovelocity_multilineage_data_path, trained_multilineage_data_path, 
                      adata_multilineage_dynamical_data_path, adata_cospar, ax[2], fig, color_dict)
    ax2[0].legend(
        bbox_to_anchor=[2.3, -0.03], ncol=4, prop={"size": 7}, fontsize=7, frameon=False
    )

    for a, label, title in zip(
        [ax[0][0], ax[1][0], ax[2][0], ax2[0]],
        ["a", "b", "c", "d"],
        ["Monocyte lineage", "Neutrophil lineage", "Bifurcation lineages", "All lineages"],
    ):
        a.text(
            -0.1,
            1.15,
            label,
            transform=a.transAxes,
            fontsize=7,
            fontweight="bold",
            va="top",
            ha="right",
        )
        a.text(
            -0.1,
            0.42,
            title,
            transform=a.transAxes,
            fontsize=7,
            fontweight="bold",
            rotation="vertical",
            va="center",
        )
        a.axis("off")


    for ext in ["", ".png"]:
        fig.savefig(
            f"{figure3_plot_name}{ext}",
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
