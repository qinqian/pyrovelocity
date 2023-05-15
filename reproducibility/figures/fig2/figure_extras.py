import errno
import os
import pickle
from logging import Logger
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from astropy import units as u
from astropy.stats import circstd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib_venn import venn2
from omegaconf import DictConfig
from statannotations.Annotator import Annotator

from pyrovelocity.config import print_config_tree
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot
from pyrovelocity.utils import get_pylogger


"""Loads trained figure 2 data and produces extra plots

Inputs:
  models:
    "models/pancreas_model1/pyrovelocity.pkl"
    "models/pbmc68k_model1/pyrovelocity.pkl"

Outputs:
  figures:
    "reports/fig2/fig2_pancreas_pbmc_uncertainties_comparison.pdf"
"""


def plots(conf: DictConfig, logger: Logger) -> None:
    """Construct summary plots for each data set and model.

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """

    time_cov_list = []
    mag_cov_list = []
    umap_mag_cov_list = []
    angle_cov_list = []
    pca_angle_cov_list = []
    pca_mag_cov_list = []
    names = []
    for data_model in conf.reports.model_summary.summarize:
        ##################
        # load data
        ##################
        print(data_model)
        data_model_conf = conf.model_training[data_model]
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path

        with open(pyrovelocity_data_path, "rb") as f:
            posterior_samples = pickle.load(f)

        cell_angles = posterior_samples["embeds_angle"] / np.pi * 180
        cell_angles_mean = cell_angles.mean(axis=0)
        angles_std = circstd(cell_angles * u.deg, method="angular", axis=0)
        cell_angles_cov = angles_std / cell_angles_mean
        angle_cov_list.append(cell_angles_cov)

        pca_cell_vector = posterior_samples["pca_vector_field_posterior_samples"]
        # (samples, cell, 50pcs)
        pca_cell_magnitudes = np.sqrt(
            (pca_cell_vector ** 2).sum(axis=-1)
        )
        pca_cell_magnitudes_mean = pca_cell_magnitudes.mean(axis=-2)
        pca_cell_magnitudes_std = pca_cell_magnitudes.std(axis=-2)
        pca_cell_magnitudes_cov = pca_cell_magnitudes_std / pca_cell_magnitudes_mean
        pca_mag_cov_list.append(pca_cell_magnitudes_cov)

        pca_cell_angles = posterior_samples["pca_embeds_angle"] / np.pi * 180
        pca_cell_angles_mean = pca_cell_angles.mean(axis=0)
        pca_angles_std = circstd(pca_cell_angles * u.deg, method="angular", axis=0)
        pca_cell_angles_cov = pca_angles_std / pca_cell_angles_mean
        pca_angle_cov_list.append(pca_cell_angles_cov)

        umap_cell_magnitudes = np.sqrt(
            (posterior_samples["vector_field_posterior_samples"] ** 2).sum(axis=-1)
        )
        umap_cell_magnitudes_mean = umap_cell_magnitudes.mean(axis=-2)
        umap_cell_magnitudes_std = umap_cell_magnitudes.std(axis=-2)
        umap_cell_magnitudes_cov = umap_cell_magnitudes_std / umap_cell_magnitudes_mean
        
        print(posterior_samples.keys())
        cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
        cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
        cell_magnitudes_std = cell_magnitudes.std(axis=-2)
        cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean

        cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
        cell_time_std = posterior_samples["cell_time"].std(0).flatten()
        cell_time_cov = cell_time_std / cell_time_mean
        time_cov_list.append(cell_time_cov)
        mag_cov_list.append(cell_magnitudes_cov)
        umap_mag_cov_list.append(umap_cell_magnitudes_cov)
        names += [data_model] * len(cell_time_cov)

    print(posterior_samples["pca_vector_field_posterior_samples"].shape)
    print(posterior_samples["embeds_angle"].shape)
    time_cov_list = np.hstack(time_cov_list)
    mag_cov_list = np.hstack(mag_cov_list)
    angle_cov_list = np.hstack(angle_cov_list)
    umap_mag_cov_list = np.hstack(umap_mag_cov_list)
    pca_angle_cov_list = np.hstack(pca_angle_cov_list)
    pca_mag_cov_list = np.hstack(pca_mag_cov_list)

    metrics_df = pd.DataFrame(
        {
            "time_coefficient_of_variation": time_cov_list,
            "magnitude_coefficient_of_variation": mag_cov_list,
            "pca_magnitude_coefficient_of_variation": pca_mag_cov_list,
            "pca_angle_coefficient_of_variation": pca_angle_cov_list,
            "umap_magnitude_coefficient_of_variation": umap_mag_cov_list,
            "umap_angle_coefficient_of_variation": angle_cov_list,
            "dataset": names,
        }
    )
    logger.info(metrics_df.head())
    shared_time_plot = conf.reports.figure2_extras.shared_time_plot
    fig, ax = plt.subplots(2, 3)
    ax = ax.flatten()
    fig.set_size_inches(15.6, 9)
    order = ("pancreas_model2", "pbmc68k_model2")
    sns.boxplot(
        x="dataset",
        y="time_coefficient_of_variation",
        data=metrics_df,
        ax=ax[0],
        order=order,
    )
    sns.boxplot(
        x="dataset",
        y="magnitude_coefficient_of_variation",
        data=metrics_df,
        ax=ax[1],
        order=order,
    )
    sns.boxplot(
        x="dataset",
        y="pca_magnitude_coefficient_of_variation", 
        data=metrics_df,
        ax=ax[2],
        order=order,
    )
    sns.boxplot(
        x="dataset",
        y="umap_magnitude_coefficient_of_variation",
        data=metrics_df,
        ax=ax[3],
        order=order,
    )
    sns.boxplot(
        x="dataset",
        y="pca_angle_coefficient_of_variation",
        data=metrics_df,
        ax=ax[4],
        order=order,
    )
    sns.boxplot(
        x="dataset",
        y="umap_angle_coefficient_of_variation",
        data=metrics_df,
        ax=ax[5],
        order=order,
    )
    pairs = [("pancreas_model2", "pbmc68k_model2")]
    time_annotator = Annotator(
        ax[0],
        pairs,
        data=metrics_df,
        x="dataset",
        y="time_coefficient_of_variation",
        order=order,
    )
    time_annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
    time_annotator.apply_and_annotate()
    mag_annotator = Annotator(
        ax[1],
        pairs,
        data=metrics_df,
        x="dataset",
        y="magnitude_coefficient_of_variation",
        order=order,
    )
    mag_annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
    mag_annotator.apply_and_annotate()

    mag_annotator = Annotator(
        ax[2],
        pairs,
        data=metrics_df,
        x="dataset",
        y="pca_magnitude_coefficient_of_variation",
        order=order,
    )
    mag_annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
    mag_annotator.apply_and_annotate()

    mag_annotator = Annotator(
        ax[3],
        pairs,
        data=metrics_df,
        x="dataset",
        y="umap_magnitude_coefficient_of_variation",
        order=order,
    )
    mag_annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
    mag_annotator.apply_and_annotate()

    # angle_annotator = Annotator(
    #    ax[2], pairs, data=metrics_df, x="dataset", y="angle_coefficient_of_variation", order=order
    # )
    # angle_annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
    # angle_annotator.apply_and_annotate()
    ax[4].set_ylim(-0.1, 0.1)
    ax[5].set_ylim(-0.1, 0.1)

    fig.savefig(
        shared_time_plot,
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
    print_config_tree(conf, logger, ())

    logger.info(
        f"\n\nVerifying existence of paths for:\n\n"
        f"  reports: {conf.reports.figure2.path}\n"
    )
    Path(conf.reports.figure2.path).mkdir(parents=True, exist_ok=True)

    print(conf.reports.figure2_extras.shared_time_plot)
    if os.path.isfile(conf.reports.figure2_extras.shared_time_plot):
        logger.info(
            f"\n\nFigure 2 outputs already exist:\n\n"
            f"  see contents of: {conf.reports.figure2.path}\n"
        )
    else:
        logger.info(f"\n\nPlotting figure 2\n\n")
        plots(conf, logger)


if __name__ == "__main__":
    main()
