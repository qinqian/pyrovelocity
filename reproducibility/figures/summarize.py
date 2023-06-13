import os
import torch
from torch.nn.functional import relu
from torch.nn.functional import softplus
import pyro
import pyro.poutine as poutine
from pyro.distributions import Bernoulli
from pyro.primitives import plate
from pyro.distributions import Beta
from pyro.distributions import LogNormal
from pyro.distributions import NegativeBinomial
from pyro.distributions import Normal
from pyro.distributions import Poisson
from pyro.distributions.constraints import positive
import pickle
from logging import Logger
from pathlib import Path
from statistics import harmonic_mean
from typing import Text

import anndata
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from astropy import units as u
from astropy.stats import circstd as acircstd
from omegaconf import DictConfig

# from scipy.stats import circmean
# from scipy.stats import circstd
from scipy.stats import circvar
from statannotations.Annotator import Annotator

from pyrovelocity.config import print_config_tree
from pyrovelocity._velocity import PyroVelocity
from pyrovelocity.data import load_data
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plot import compute_mean_vector_field
from pyrovelocity.plot import compute_volcano_data
from pyrovelocity.plot import get_posterior_sample_angle_uncertainty
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot
from pyrovelocity.plot import us_rainbowplot
from pyrovelocity.plot import vector_field_uncertainty
from pyrovelocity.plots.rainbow import pareto_frontier_genes
from pyrovelocity.utils import anndata_counts_to_df, mRNA
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import mae_evaluate
from pyrovelocity.utils import print_anndata
from pyrovelocity.utils import print_attributes


"""Loads model-trained data and generates figures.

for data_model in
    [
        pancreas_model1,
        pancreas_model2,
        pbmc68k_model1,
        pbmc68k_model2,
        pons_model1,
        pons_model2,
    ]

Inputs:
  data:
    "models/{data_model}/trained.h5ad"
    "models/{data_model}/pyrovelocity.pkl"

Outputs:
  figures:
    shared_time_plot: reports/{data_model}/shared_time.pdf
    volcano_plot: reports/{data_model}/volcano.pdf
    rainbow_plot: reports/{data_model}/rainbow.pdf
    uncertainty_param_plot: reports/{data_model}/param_uncertainties.pdf
    vector_field_plot: reports/{data_model}/vector_field.pdf
    biomarker_selection_plot: reports/{data_model}/markers_selection_scatterplot.tif
    biomarker_phaseportrait_plot: reports/{data_model}/markers_phaseportrait.pdf
"""


def summarize_fig2_part1(
    adata,
    posterior_vector_field,
    posterior_time,
    cell_magnitudes,
    pca_embeds_angle,
    embed_radians,
    embedding,
    embed_mean,
    cluster="cell_type",
    plot_name="test",
):
    dot_size = 3.5
    font_size = 6.5
    scale = 0.35
    scale_high = 7.8
    scale_low = 7.8

    arrow = 3.6
    density = 0.4
    ress = pd.DataFrame(
        {
            "cell_type": adata.obs[cluster].values,
            "X1": adata.obsm[f"X_{embedding}"][:, 0],
            "X2": adata.obsm[f"X_{embedding}"][:, 1],
        }
    )
    fig = plt.figure(figsize=(9.6, 2), constrained_layout=False)
    fig.subplots_adjust(
        hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.45
    )
    ax = fig.subplots(1, 6)
    pos = ax[0].get_position()

    sns.scatterplot(
        x="X1",
        y="X2",
        data=ress,
        alpha=0.9,
        s=dot_size,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        ax=ax[0],
        legend="brief",
    )
    ax[0].axis("off")
    ax[0].set_title("Cell types\n", fontsize=font_size)
    print(pos.x0, pos.x1)
    ax[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.5, -0.48),
        ncol=5,
        fancybox=True,
        prop={"size": font_size},
        fontsize=font_size,
        frameon=False,
    )
    kwargs = dict(
        color="gray",
        s=dot_size,
        show=False,
        alpha=0.25,
        min_mass=3.5,
        scale=scale,
        frameon=False,
        density=density,
        arrow_size=3,
        linewidth=1,
    )
    scv.pl.velocity_embedding_grid(
        adata, basis=embedding, fontsize=font_size, ax=ax[1], title="", **kwargs
    )
    ax[1].set_title("Scvelo\n", fontsize=7)
    scv.pl.velocity_embedding_grid(
        adata,
        fontsize=font_size,
        basis=embedding,
        title="",
        ax=ax[2],
        vkey="velocity_pyro",
        **kwargs,
    )
    ax[2].set_title("Pyro-Velocity\n", fontsize=7)

    pca_cell_angles = pca_embeds_angle / np.pi * 180  # degree
    pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)

    cell_time_mean = posterior_time.mean(0).flatten()
    cell_time_std = posterior_time.std(0).flatten()
    cell_time_cov = cell_time_std / cell_time_mean

    plot_vector_field_uncertain(
        adata,
        embed_mean,
        cell_time_std,
        ax=ax[3],
        cbar=True,
        fig=fig,
        basis=embedding,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="shared time",
        cmap="winter",
        cmax=None,
    )

    cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
    cell_magnitudes_std = cell_magnitudes.std(axis=-2)
    cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
    plot_vector_field_uncertain(
        adata,
        embed_mean,
        cell_magnitudes_cov,
        ax=ax[4],
        cbar=True,
        fig=fig,
        basis=embedding,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="base magnitude",
        cmap="summer",
        cmax=None,
    )

    plot_vector_field_uncertain(
        adata,
        embed_mean,
        pca_angles_std,
        ax=ax[5],
        cbar=True,
        fig=fig,
        basis=embedding,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="PCA angle",
        cmap="inferno",
        cmax=None,
    )
    for ext in ["", ".png"]:
        fig.savefig(
            f"{plot_name}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def summarize_fig2_part2(
    adata, posterior_samples, plot_name="", basis="", cell_state="", fig=None
):
    if fig is None:
        fig = plt.figure(figsize=(9.5, 5))
        subfigs = fig.subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[1.8, 4])
        ax = subfigs[0].subplots(2, 1)
        plot_posterior_time(
            posterior_samples,
            adata,
            ax=ax[0],
            fig=subfigs[0],
            addition=False,
            basis=basis,
        )
        volcano_data, _ = plot_gene_ranking(
            [posterior_samples], [adata], ax=ax[1], time_correlation_with="st"
        )
        _ = rainbowplot(
            volcano_data,
            adata,
            posterior_samples,
            subfigs[1],
            data=["st", "ut"],
            basis=basis,
            cell_state=cell_state,
            num_genes=4,
        )
        for ext in ["", ".png"]:
            fig.savefig(
                f"{plot_name}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )


def cluster_violin_plots(
    logger: Logger,
    data_model: str,
    adata: anndata.AnnData,
    posterior_samples,
    cluster_key: str,
    violin_flag: bool,
    pairs: list,
    show_outlier: bool,
    fig_name: str = None,
) -> None:
    """Construct violin plots for each cluster."""

    time_cov_list = []
    mag_cov_list = []
    umap_mag_cov_list = []
    umap_angle_std_list = []
    pca_angle_std_list = []
    pca_mag_cov_list = []
    pca_angle_uncertain_list = []
    umap_angle_uncertain_list = []
    names = []

    # get cluster order
    cluster_time_list = []
    clusters = adata.obs[cluster_key].values.categories
    for cluster in clusters:
        adata_cluster = adata[adata.obs[cluster_key] == cluster]
        cluster_time = adata_cluster.obs["velocity_pseudotime"].mean()
        cluster_time_list.append(cluster_time)
    sorted_cluster_id = sorted(
        range(len(cluster_time_list)), key=lambda k: cluster_time_list[k], reverse=False
    )
    order = clusters[sorted_cluster_id]

    umap_cell_angles = posterior_samples["embeds_angle"] / np.pi * 180
    # umap_cell_cirsvar = circvar(umap_cell_angles, axis=0)
    umap_angle_std = acircstd(umap_cell_angles * u.deg, method="angular", axis=0)
    umap_angle_std_list.append(umap_angle_std)
    umap_angle_uncertain = get_posterior_sample_angle_uncertainty(umap_cell_angles)
    umap_angle_uncertain_list.append(umap_angle_uncertain)

    pca_cell_vector = posterior_samples["pca_vector_field_posterior_samples"]
    pca_cell_magnitudes = np.sqrt((pca_cell_vector**2).sum(axis=-1))
    pca_cell_magnitudes_mean = pca_cell_magnitudes.mean(axis=-2)
    pca_cell_magnitudes_std = pca_cell_magnitudes.std(axis=-2)
    pca_cell_magnitudes_cov = pca_cell_magnitudes_std / pca_cell_magnitudes_mean
    pca_mag_cov_list.append(pca_cell_magnitudes_cov)

    pca_cell_angles = posterior_samples["pca_embeds_angle"] / np.pi * 180
    # pca_cell_cirsvar = circvar(pca_cell_angles, axis=0)
    pca_cell_cirsstd = acircstd(pca_cell_angles * u.deg, method="angular", axis=0)
    pca_angle_std_list.append(pca_cell_cirsstd)
    pca_angle_uncertain = get_posterior_sample_angle_uncertainty(pca_cell_angles)
    pca_angle_uncertain_list.append(pca_angle_uncertain)

    umap_cell_magnitudes = np.sqrt(
        (posterior_samples["vector_field_posterior_samples"] ** 2).sum(axis=-1)
    )
    umap_cell_magnitudes_mean = umap_cell_magnitudes.mean(axis=-2)
    umap_cell_magnitudes_std = umap_cell_magnitudes.std(axis=-2)
    umap_cell_magnitudes_cov = umap_cell_magnitudes_std / umap_cell_magnitudes_mean

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
    name = list(adata.obs[cluster_key])
    names += name

    time_cov_list = np.hstack(time_cov_list)
    mag_cov_list = np.hstack(mag_cov_list)
    pca_mag_cov_list = np.hstack(pca_mag_cov_list)
    pca_angle_std_list = np.hstack(pca_angle_std_list)
    umap_mag_cov_list = np.hstack(umap_mag_cov_list)
    umap_angle_std_list = np.hstack(umap_angle_std_list)

    metrics_df = pd.DataFrame(
        {
            r"$CoV({\mathrm{time}})$": time_cov_list,
            r"$CoV({\mathrm{magnitude}})$": mag_cov_list,
            r"$Std({\mathrm{angle}}_{pca})$": pca_angle_std_list,
            r"$CoV({\mathrm{magnitude}}_{pca})$": pca_mag_cov_list,
            r"$Std({\mathrm{angle}}_{umap})$": umap_angle_std_list,
            r"$CoV({\mathrm{magnitude}}_{umap})$": umap_mag_cov_list,
            "dataset": names,
        }
    )

    max_values, min_values = {}, {}
    for key in metrics_df.keys()[0:6]:
        key_data = metrics_df[key]
        q1, q3 = np.percentile(key_data, (25, 75))
        max_values[key] = q3 + (q3 - q1) * 1.5
        if key_data.min() >= 0:
            min_values[key] = 0
        else:
            min_values[key] = q1 - (q3 - q1) * 1.5
    print(max_values)
    print(min_values)

    if "log" in fig_name:
        log_time_cov_list = np.log(time_cov_list)
        log_mag_cov_list = np.log(mag_cov_list)
        log_umap_mag_cov_list = np.log(umap_mag_cov_list)
        pca_angle_uncertain_list = np.hstack(pca_angle_uncertain_list)
        log_pca_mag_cov_list = np.log(pca_mag_cov_list)
        umap_angle_uncertain_list = np.hstack(umap_angle_uncertain_list)
        metrics_df = pd.DataFrame(
            {
                r"$\log(CoV({\mathrm{time}}))$": log_time_cov_list,
                r"$\log(CoV({\mathrm{magnitude}}))$": log_mag_cov_list,
                r"$CircStd({\mathrm{angle}}_{pca})$": pca_angle_uncertain_list,
                r"$\log(CoV({\mathrm{magnitude}}_{pca}))$": log_pca_mag_cov_list,
                r"$CircStd({\mathrm{angle}}_{umap})$": umap_angle_uncertain_list,
                r"$\log(CoV({\mathrm{magnitude}}_{umap}))$": log_umap_mag_cov_list,
                "dataset": names,
            }
        )

    logger.info(metrics_df.head())
    parameters = {"axes.labelsize": 25, "axes.titlesize": 35}
    plt.rcParams.update(parameters)
    fig, ax = plt.subplots(6, 1)
    ax = ax.flatten()
    fig.set_size_inches(20, 60)
    ax[0].set_title(f"{data_model.split('_')[0]}")
    # ax[0].set_ylim(-5.5, -2)

    if violin_flag:
        for i in range(6):
            sns.violinplot(
                x="dataset",
                y=metrics_df.keys()[i],
                data=metrics_df,
                ax=ax[i],
                order=order,
                showfliers=show_outlier,
            )
    else:
        for i in range(6):
            sns.boxplot(
                x="dataset",
                y=metrics_df.keys()[i],
                data=metrics_df,
                ax=ax[i],
                order=order,
                showfliers=show_outlier,
            )

    if not pairs is None:
        for i in range(6):
            annotator = Annotator(
                ax[i],
                pairs,
                data=metrics_df,
                x="dataset",
                y=metrics_df.keys()[i],
                order=order,
            )
            annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
            annotator.apply_and_annotate()

    for axi in ax:
        axi.tick_params(axis="both", labelsize=20)
        axi.set_xlabel("")

    for ext in ["", ".png"]:
        fig.savefig(
            f"{fig_name}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def plot_parameter_posterior_distributions(
    posterior_samples,
    adata: anndata.AnnData,
    geneset,
    parameter_uncertainty_plot_path: str,
):
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(18, 12)
    for index, kinetics in enumerate(["alpha", "beta", "gamma"]):
        print(posterior_samples[kinetics].squeeze().shape)
        print(np.isin(adata.var_names, list(geneset)).sum())
        df = pd.DataFrame(
            np.log(
                posterior_samples[kinetics].squeeze()[
                    :, np.isin(adata.var_names, list(geneset))
                ],
            ),
            columns=adata.var_names[np.isin(adata.var_names, list(geneset))],
        )
        df = df.apply(lambda x: x - x.mean())
        df_long = df.melt(var_name="index", value_name="value")
        print(df_long.head())
        df_long["index"] = pd.Categorical(
            df_long["index"], categories=geneset, ordered=True
        )
        ax1 = sns.violinplot(
            x="index", y="value", data=df_long, ax=ax[index], showfliers=False
        )
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")
        ax1.set_ylabel(kinetics)
        ax1.set_xlabel("")
    fig.subplots_adjust(
        hspace=0.4, wspace=0.45, left=0.08, right=0.95, top=0.9, bottom=0.15
    )
    for ext in ["", ".png"]:
        fig.savefig(
            f"{parameter_uncertainty_plot_path}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def extended_time_model(u_obs, s_obs, *, expand_time_span: int = 1000):
    """ a sampler that use all the parameters of pyrovelocity 
    except the cell_time, we replaced the cell_time with a 
    grid points of time.
    """
    expanded_time_plate = pyro.plate("new_grid_cells", u_obs.shape[0], dim=-2)
    gene_plate = pyro.plate("genes", 2000, dim=-1)
    with gene_plate, poutine.mask(mask=True):
        alpha = pyro.sample("alpha", LogNormal(0, 1))
        gamma = pyro.sample("gamma", LogNormal(0, 1))
        beta = pyro.sample("beta", LogNormal(0, 1))
        u0 = pyro.sample("u_offset", LogNormal(0, 1))
        s0 = pyro.sample("s_offset", LogNormal(0, 1))
        t0 = pyro.sample("t0", Normal(0, 1))
        u_scale = pyro.sample("u_scale", LogNormal(0, 0.1))
        s_scale = u_scale.new_ones(1)
        dt_switching = pyro.sample("dt_switching", LogNormal(0, 0.1))
        u_inf, s_inf = mRNA(dt_switching, u0, s0, alpha, beta, gamma)
        u_inf = pyro.deterministic("u_inf", u_inf, event_dim=0)
        s_inf = pyro.deterministic("s_inf", s_inf, event_dim=0)
        print(u_inf, s_inf)
        print(u0, s0)
        switching = pyro.deterministic(
            "switching", dt_switching + t0, event_dim=0
        )

    with expanded_time_plate:
        # turn shared time to be deterministic
        t = pyro.deterministic("expanded_cell_time", torch.linspace(-expand_time_span, expand_time_span, expanded_time_plate.size))
        t = t.unsqueeze(-1)
        print(t)
        print(t.shape)
        with gene_plate:
            state = (
                pyro.sample(
                    "cell_gene_state",
                    Bernoulli(logits=t - switching),
                    infer={"enumerate": 'parallel'},
                )
                == 0
            )
            print(state)
            alpha_off = alpha.new_zeros(1)
            u0_vec = torch.where(state, u0, u_inf)
            s0_vec = torch.where(state, s0, s_inf)
            alpha_vec = torch.where(state, alpha, alpha_off)
            tau = softplus(torch.where(state, t - t0, t - switching))
            ut, st = mRNA(tau, u0_vec, s0_vec, alpha_vec, beta, gamma)
            ut = ut * u_scale / s_scale
            print(ut)
            print(st)
            ut = relu(ut) + 1e-6
            st = relu(st) + 1e-6
            print(ut)
            print(st)
            ut = pyro.deterministic("ut", ut, event_dim=0)
            st = pyro.deterministic("st", st, event_dim=0)
    return ut, st


def extrapolate_time_phase_portrait_curve():
    adata = scv.read(trained_data_path)
    pyro.clear_param_store()
    PyroVelocity.setup_anndata(adata)
    model = PyroVelocity(adata)
    print(pyrovelocity_model_path)
    model = model.load_model(pyrovelocity_model_path, adata, use_gpu=0)
    from pyro.infer import MCMC, NUTS, Predictive

    predictive_svi = Predictive(extended_time_model, guide=model._guide, num_samples=20)()
    ###posterior_samples = model.generate_posterior_samples(model.adata, num_samples=21)
    #print(posterior_samples.keys())
    #print(posterior_samples["st"].shape)
    #print(posterior_samples["ut"].shape)
    ##for figi, gene in enumerate(['Iapp', 'Cpe', 'Pcsk2', 'Tmem27', 'Ins1', 'Ins2']):
    #for figi, gene in enumerate(model.adata.var_names[:5]):
    #    (index,) = np.where(adata.var_names == gene)
    #    fig, ax = plt.subplots(4, 5)
    #    fig.set_size_inches(18, 12)
    #    ax = ax.flatten()
    #    for sample in range(20):
    #        ax[sample].scatter(posterior_samples["st"][sample][:,index[0]], 
    #                           posterior_samples["ut"][sample][:,index[0]], s=1.5, linewidth=0, color='r')
    #        ax[sample].set_title(f"{gene} model 2 sample {sample}")
    #        ax[sample].set_xlim(0, np.max(posterior_samples["st"][:, :, index[0]])*1.1)
    #        ax[sample].set_ylim(0, np.max(posterior_samples["ut"][:, :, index[0]])*1.1)
    #    fig.tight_layout()
    #    fig.savefig(
    #        f"fig{figi}_test.png",
    #        facecolor=fig.get_facecolor(),
    #        bbox_inches="tight",
    #        edgecolor="none",
    #        dpi=300,
    #    )
    #return


def plots(conf: DictConfig, logger: Logger) -> None:
    """Construct summary plots for each data set and model.

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """
    for data_model in conf.train_models:
        ##################
        # load data
        ##################

        data_model_conf = conf.model_training[data_model]
        cell_state = data_model_conf.training_parameters.cell_state
        trained_data_path = data_model_conf.trained_data_path
        print(trained_data_path)
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        posterior_samples_data_path = data_model_conf.posterior_samples_path

        reports_data_model_conf = conf.reports.model_summary[data_model]
        trained_data_path = reports_data_model_conf.trained_data_path
        pyrovelocity_data_path = reports_data_model_conf.pyrovelocity_data_path

        # print_config_tree(reports_data_model_conf, logger, ())

        logger.info(f"\n\nPlotting summary figure(s) in: {data_model}\n\n")

        logger.info(
            f"\n\nVerifying existence of path for:\n\n"
            f"  reports: {reports_data_model_conf.path}\n"
        )
        Path(reports_data_model_conf.path).mkdir(parents=True, exist_ok=True)

        dataframe_path = reports_data_model_conf.dataframe_path
        volcano_plot = reports_data_model_conf.volcano_plot
        rainbow_plot = reports_data_model_conf.rainbow_plot
        vector_field_plot = reports_data_model_conf.vector_field_plot
        shared_time_plot = reports_data_model_conf.shared_time_plot
        fig2_part1_plot = reports_data_model_conf.fig2_part1_plot
        fig2_part2_plot = reports_data_model_conf.fig2_part2_plot
        violin_clusters_lin = reports_data_model_conf.violin_clusters_lin
        violin_clusters_log = reports_data_model_conf.violin_clusters_log
        parameter_uncertainty_plot_path = reports_data_model_conf.uncertainty_param_plot

        output_filenames = [
            dataframe_path,
            volcano_plot,
            rainbow_plot,
            vector_field_plot,
            shared_time_plot,
            parameter_uncertainty_plot_path,
        ]
        if all(os.path.isfile(f) for f in output_filenames):
            logger.info(
                "\n\t" + "\n\t".join(output_filenames) + "\nAll output files exist"
            )
            return logger.warn("Remove output files if you want to regenerate them.")

        logger.info(f"Loading trained data: {trained_data_path}")
        adata = scv.read(trained_data_path)
        # gene_mapping = {"1100001G20Rik": "Wfdc21"}
        # adata = rename_anndata_genes(adata, gene_mapping)

        pyrovelocity_model_path = data_model_conf.model_path
        print(pyrovelocity_model_path)
        PyroVelocity.setup_anndata(adata)
        model = PyroVelocity(adata)
        print(pyrovelocity_model_path)
        model = model.load_model(pyrovelocity_model_path, adata, use_gpu=0)
        posterior_samples = model.generate_posterior_samples(model.adata, num_samples=21)
        print(posterior_samples.keys())
        print(posterior_samples["st"].shape)
        print(posterior_samples["ut"].shape)
        #for figi, gene in enumerate(['Iapp', 'Cpe', 'Pcsk2', 'Tmem27', 'Ins1', 'Ins2']):
        for figi, gene in enumerate(model.adata.var_names[:5]):
            (index,) = np.where(adata.var_names == gene)
            fig, ax = plt.subplots(4, 5)
            fig.set_size_inches(18, 12)
            ax = ax.flatten()
            for sample in range(20):
                ax[sample].scatter(posterior_samples["st"][sample][:,index[0]], 
                                   posterior_samples["ut"][sample][:,index[0]], s=1.5, linewidth=0, color='r')
                ax[sample].set_title(f"{gene} model 2 sample {sample}")
                ax[sample].set_xlim(0, np.max(posterior_samples["st"][:, :, index[0]])*1.1)
                ax[sample].set_ylim(0, np.max(posterior_samples["ut"][:, :, index[0]])*1.1)
            fig.tight_layout()
            fig.savefig(
                f"fig{figi}_test.png",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )

        logger.info(f"Loading pyrovelocity data: {pyrovelocity_data_path}")
        posterior_samples = CompressedPickle.load(pyrovelocity_data_path)
        #all_posterior_samples = CompressedPickle.load(posterior_samples_data_path)
        #print(all_posterior_samples.keys())

        ##################
        # save dataframe
        ##################

        logger.info(f"Saving AnnData object to dataframe: {dataframe_path}")
        (
            df,
            total_obs,
            total_var,
            max_spliced,
            max_unspliced,
        ) = anndata_counts_to_df(adata)

        CompressedPickle.save(
            dataframe_path,
            (
                df,
                total_obs,
                total_var,
                max_spliced,
                max_unspliced,
            ),
        )
        print(posterior_samples.keys())

        vector_field_basis = data_model_conf.vector_field_parameters.basis

        cell_type = data_model_conf.training_parameters.cell_state

        ##################
        # generate figures
        ##################

        # vector fields
        if os.path.isfile(fig2_part1_plot):
            logger.info(f"{fig2_part1_plot} exists")
        else:
            logger.info(f"Generating figure: {fig2_part1_plot}")
            summarize_fig2_part1(
                adata,
                posterior_samples["vector_field_posterior_samples"],
                posterior_samples["cell_time"],
                posterior_samples["original_spaces_embeds_magnitude"],
                posterior_samples["pca_embeds_angle"],
                posterior_samples["embeds_angle"],
                vector_field_basis,
                posterior_samples["vector_field_posterior_mean"],
                cell_type,
                fig2_part1_plot,
            )

        # gene selection
        if os.path.isfile(fig2_part2_plot):
            logger.info(f"{fig2_part2_plot} exists")
        else:
            logger.info(f"Generating figure: {fig2_part2_plot}")
            summarize_fig2_part2(
                adata,
                posterior_samples,
                basis=vector_field_basis,
                cell_state=cell_type,
                plot_name=fig2_part2_plot,
                fig=None,
            )

        # cluster violin plots
        if os.path.isfile(violin_clusters_log):
            logger.info(f"{violin_clusters_log} exists")
        else:
            logger.info(f"Generating figure: {violin_clusters_log}")
            for fig_name in [violin_clusters_lin, violin_clusters_log]:
                cluster_violin_plots(
                    logger,
                    data_model,
                    adata=adata,
                    posterior_samples=posterior_samples,
                    cluster_key=cell_state,
                    violin_flag=True,
                    pairs=None,
                    show_outlier=False,
                    fig_name=fig_name,
                )

        # shared time plot
        cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
        cell_time_std = posterior_samples["cell_time"].std(0).flatten()
        adata.obs["shared_time_uncertain"] = cell_time_std
        adata.obs["shared_time_mean"] = cell_time_mean

        if os.path.isfile(shared_time_plot):
            logger.info(f"{shared_time_plot} exists")
        else:
            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(9.2, 2.6)
            ax = ax.flatten()
            ax_cb = scv.pl.scatter(
                adata,
                c="shared_time_mean",
                ax=ax[0],
                show=False,
                cmap="inferno",
                fontsize=7,
                colorbar=True,
            )
            ax_cb = scv.pl.scatter(
                adata,
                c="shared_time_uncertain",
                ax=ax[1],
                show=False,
                cmap="inferno",
                fontsize=7,
                colorbar=True,
            )
            select = adata.obs["shared_time_uncertain"] > np.quantile(
                adata.obs["shared_time_uncertain"], 0.9
            )
            sns.kdeplot(
                adata.obsm[f"X_{vector_field_basis}"][:, 0][select],
                adata.obsm[f"X_{vector_field_basis}"][:, 1][select],
                ax=ax[1],
                levels=3,
                fill=False,
            )
            ax[2].hist(cell_time_std / cell_time_mean, bins=100)
            fig.savefig(
                shared_time_plot,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )

        volcano_data = posterior_samples["gene_ranking"]
        number_of_marker_genes = min(
            max(int(len(volcano_data) * 0.1), 4), 20, len(volcano_data)
        )
        print(f"Searching for {number_of_marker_genes} marker genes")
        geneset = pareto_frontier_genes(volcano_data, number_of_marker_genes)

        # volcano plot
        if os.path.isfile(volcano_plot):
            logger.info(f"{volcano_plot} exists")
        else:
            logger.info(f"Generating figure: {volcano_plot}")

            volcano_data, fig = plot_gene_ranking(
                [posterior_samples],
                [adata],
                selected_genes=geneset,
                time_correlation_with="st",
                show_marginal_histograms=True,
            )

            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            for ext in ["", ".png"]:
                fig.savefig(
                    f"{volcano_plot}{ext}",
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                    edgecolor="none",
                    dpi=300,
                )

        # parameter uncertainty
        if os.path.isfile(parameter_uncertainty_plot_path):
            logger.info(f"{parameter_uncertainty_plot_path} exists")
        else:
            logger.info(f"Generating figure: {parameter_uncertainty_plot_path}")
            plot_parameter_posterior_distributions(
                posterior_samples=posterior_samples,
                adata=adata,
                geneset=geneset,
                parameter_uncertainty_plot_path=parameter_uncertainty_plot_path,
            )

        # rainbow plot
        from pyrovelocity.plots.rainbow import rainbowplot

        if os.path.isfile(rainbow_plot):
            logger.info(f"{rainbow_plot} exists")
        else:
            logger.info(f"Generating figure: {rainbow_plot}")
            fig = rainbowplot(
                volcano_data=volcano_data,
                adata=adata,
                posterior_samples=posterior_samples,
                genes=geneset,
                data=["st", "ut"],
                basis=vector_field_basis,
                cell_state=cell_state,
            )

            for ext in ["", ".png"]:
                fig.savefig(
                    f"{rainbow_plot}{ext}",
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                    edgecolor="none",
                    dpi=300,
                )

        # mean vector field plot
        if os.path.isfile(vector_field_plot):
            logger.info(f"{vector_field_plot} exists")
        else:
            logger.info(f"Generating figure: {vector_field_plot}")
            fig, ax = plt.subplots()

            scv.pl.velocity_embedding_grid(
                adata,
                basis=vector_field_basis,
                color=cell_state,
                title="",
                vkey="velocity_pyro",
                linewidth=1,
                ax=ax,
                show=False,
                legend_loc="right margin",
                density=0.4,
                scale=0.2,
                arrow_size=2,
                arrow_length=2,
                arrow_color="black",
            )
            for ext in ["", ".png"]:
                fig.savefig(
                    f"{vector_field_plot}{ext}",
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                    edgecolor="none",
                    dpi=300,
                )


def rename_anndata_genes(adata, gene_mapping):
    """
    Renames genes in an AnnData object according to a dictionary mapping.

    Parameters:
    -----------
    adata: anndata.AnnData
        The AnnData object.
    gene_mapping: dict
        A dictionary containing mappings of old gene names (keys) to new gene names (values).

    Returns:
    -------
    adata: anndata.AnnData
        The AnnData object with renamed genes.
    """

    var_names = adata.var_names.tolist()
    var_names = [
        gene_mapping[name] if name in gene_mapping else name for name in var_names
    ]
    adata.var_names = var_names

    return adata


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Plots figures for model summary.
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)
    plots(conf, logger)


#if __name__ == "__main__":
#    main()
