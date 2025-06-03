# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Abstract
#
# TODO

# %% [markdown]
# ## Imports and setup

# %%
# #! black nb_pareto2d_demo.py

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import gzip
import json
import logging
import time
import pathlib


import numpy as np
import matplotlib.pyplot as plt

import pareto2d

# %%
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s.%(msecs)03d500: %(levelname).1s %(name)s.py:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Here you put modules where you want more verbose logging

    for module_name in [__name__, "pf2d"]:
        logging.getLogger(module_name).setLevel(logging.INFO)


setup_logging()


# %% [markdown]
# ## Helper functions
#
#


# %%
class NoGlobals:
    def __init__(self, logger=None):
        self.logger = logger

    @staticmethod
    def _get_global_ids():
        return [v for v in globals().keys() if not v.startswith("_")]

    def _keep_only_ids(self, ids):
        ids_all = list(globals().keys())
        for id_cur in ids_all:
            if not id_cur.startswith("_") and id_cur not in ids:
                if self.logger is not None:
                    self.logger.info("Deleting " + id_cur)
                del globals()[id_cur]

    def __enter__(self):
        self.globals = self._get_global_ids()

    def __exit__(self, type, value, traceback):
        self._keep_only_ids(self.globals)


# %%
class Timing:
    def __init__(self, block_name):
        self.block_name = block_name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        duration = time.perf_counter() - self.start
        logger.info(f"{self.block_name} took {duration:.2f} sec.")


# %%
def _open_rt(fname):
    if str(fname).endswith(".gz"):
        return gzip.open(fname, "rt")
    else:
        return open(fname, "rt")


def is_data_ok(di):
    if "status" in di:
        return di["status"].lower() == "ok"
    else:
        return True


def read_data(fname):
    with _open_rt(fname) as f:
        data = [json.loads(line) for line in f]
    data = [di for di in data if is_data_ok(di)]
    return data


def read_baseline(fname):
    with _open_rt(fname) as f:
        data = [json.loads(line) for line in f]
    return [d for d in data if d["type"] == "zerl"]


# %%
# def _has_duplicates(o1, o2):
#     ii = np.lexsort((o2, o1))
#     o1_sorted = o1[ii]
#     o2_sorted = o2[ii]
#     delta_o1 = o1_sorted[1:] - o1_sorted[:-1]
#     delta_o2 = o2_sorted[1:] - o2_sorted[:-1]
#     n = np.sum((delta_o1==0) & (delta_o2==0))
#     return n>0

# def remove_duplicates(o1, o2):
#     ii = np.lexsort((o2, o1))
#     o1_sorted = o1[ii]
#     o2_sorted = o2[ii]
#     delta_o1 = o1_sorted[1:] - o1_sorted[:-1]
#     delta_o2 = o2_sorted[1:] - o2_sorted[:-1]
#     non_duplicates = (delta_o1!=0) | (delta_o2!=0)
#     nn = len(o1) - np.sum(non_duplicates) -1
#     logger.info(f"Removing {nn} duplicates")
#     non_duplicates = non_duplicates.tolist()
#     non_duplicates.append(True)
#     non_duplicates = np.array(non_duplicates)
#     return o1[ii[non_duplicates]], o2[ii[non_duplicates]]


# def is_pareto(o1, o2):
#     # o1, o2 -> maximized
#     assert not _has_duplicates(o1, o2)
#     n = len(o1)
#     is_pareto = np.zeros(n, dtype=bool)
#     for i in range(n):
#         o1i = o1[i]
#         o2i = o2[i]
#         is_pareto[i] = np.sum((o1 >= o1i) & (o2 >= o2i)) <= 1

#     return is_pareto

# def get_sorted_pf(o1, o2):
#     is_pf = is_pareto(o1, o2)
#     o1_pf = o1[is_pf]
#     o2_pf = o2[is_pf]
#     ii = np.argsort(o1_pf)
#     return o1_pf[ii], o2_pf[ii]

# def _get_pareto_o2_one_value(o1, o1_pf_sorted, o2_pf_sorted):
#     i = 0
#     found = False
#     n = len(o1_pf_sorted)
#     while i<n:
#         if o1_pf_sorted[i] > o1:
#             found = True
#             break
#         i+=1
#     if found:
#         return o2_pf_sorted[i]
#     else:
#         return 0.0


# def get_pareto_o2(o1, o1_pf, o2_pf):
#     return np.fromiter((_get_pareto_o2_one_value(o1i, o1_pf, o2_pf) for o1i in o1), o1.dtype)

# %%
PARAMS_FACTOR = 1.0e3


# %% [markdown]
# ## Main - Pareto corridor by  epsilon filtering


# %%
def _():
    dpath = pathlib.Path(
        "inp_ptblop_qwen32b_mbpp/2025-03-16_qwen25-32bCO_mbpp_atcl-parf-p06u/bp_configs.json.gz"
    )
    data = read_data(dpath)

    size = np.array([d["mparams"] for d in data]) / PARAMS_FACTOR
    quality = np.array([d["evaluation"]["mbpp"] for d in data])
    size, quality = pareto2d.get_dedpuplicated_copy(size, quality)
    size, quality = pareto2d.get_dedpuplicated_copy(size, quality)
    size, quality = pareto2d.get_dedpuplicated_copy(size, quality)

    size_pf, quality_pf = pareto2d.get_pf(size, quality, pareto2d.Mode.O1_MIN_O2_MAX)
    size_plot = np.arange(22, 33, 0.01)
    quality_plot = pareto2d.get_pf_o2(
        size_plot, size_pf, quality_pf, pareto2d.Mode.O1_MIN_O2_MAX
    )
    quality_p = pareto2d.get_pf_o2(
        size, size_pf, quality_pf, pareto2d.Mode.O1_MIN_O2_MAX
    )
    # QUALITY_EPSILON = 0.05
    # mask_epsilon = np.abs(quality-quality_p) < QUALITY_EPSILON
    SIZE_EPSILON = 1
    size_p = pareto2d.get_pf_o1(
        quality, size_pf, quality_pf, pareto2d.Mode.O1_MIN_O2_MAX
    )
    mask_epsilon = np.abs(size - size_p) < SIZE_EPSILON

    ALPHA = 0.3
    cm = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(size, quality, label="test", color=cm(0), alpha=ALPHA)

    ax.scatter(size_pf, quality_pf, label="test", color=cm(2), alpha=1.0)
    ax.plot(size_plot, quality_plot, label="test", color=cm(2), alpha=1.0)
    ax.scatter(
        size[mask_epsilon], quality[mask_epsilon], label="test", color=cm(2), alpha=1.0
    )

    ax.grid()


_()


# %% [markdown]
# ## Main - creating Pareto corridor by multiple Pareto fronts


# %%
def _():
    dpath = pathlib.Path(
        "inp_ptblop_qwen32b_mbpp/2025-03-16_qwen25-32bCO_mbpp_atcl-parf-p06u/bp_configs.json.gz"
    )
    data = read_data(dpath)

    size = np.array([d["mparams"] for d in data]) / PARAMS_FACTOR
    quality = np.array([d["evaluation"]["mbpp"] for d in data])

    NPF = 6
    size_pf, quality_pf = [None] * NPF, [None] * NPF

    for i in range(NPF):
        mask = pareto2d.get_pf_mask(size, quality, pareto2d.Mode.O1_MIN_O2_MAX)
        size_pf[i], quality_pf[i] = size[mask], quality[mask]
        size, quality = size[~mask], quality[~mask]

    ALPHA = 0.3
    cm = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(size, quality, label="1", color=cm(1), alpha=ALPHA)
    for i in range(NPF):
        ax.scatter(
            size_pf[i], quality_pf[i], label=f"pf {i}", color=cm(2 + 2 * i), alpha=1.0
        )


_()


# %%
def _():
    dpath = pathlib.Path(
        "/nas/people/michal_lopuszynski/JOBS_BP6/2025-04-24_qwen25-32b_mbpp_beam/out/pareto_fronts/pareto_front_0000.json"
    )
    data = read_data(dpath)

    size = np.array([d["mparams_pred"] for d in data]) / PARAMS_FACTOR
    quality = np.array([d["mbpp_pred"] for d in data])

    NPF = 5
    size_pf, quality_pf = [None] * NPF, [None] * NPF

    for i in range(NPF):
        mask = pareto2d.get_pf_mask(size, quality, pareto2d.Mode.O1_MIN_O2_MAX)
        size_pf[i], quality_pf[i] = size[mask], quality[mask]
        size, quality = size[~mask], quality[~mask]

    ALPHA = 0.3
    cm = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(size, quality, label="1", color=cm(1), alpha=ALPHA)
    for i in range(NPF):
        ax.scatter(
            size_pf[i], quality_pf[i], label=f"pf {i}", color=cm(2 + 2 * i), alpha=1.0
        )


_()

# %%
