# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# | echo: false
import logging
import time

# %%
# | echo: false

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s.%(msecs)03d500: %(levelname).1s %(name)s.py:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Here you put modules where you want more verbose logging

    for module_name in [__name__]:
        logging.getLogger(module_name).setLevel(logging.INFO)


setup_logging()


# %%
# | echo: false


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
# | echo: false


class Timing:
    def __init__(self, block_name):
        self.block_name = block_name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        duration = time.perf_counter() - self.start
        logger.info(f"{self.block_name} took {duration:.2f} sec.")


# %%
# | echo: false


def get_pareto2d_ideal_solution(mode):
    if mode == pareto2d.Mode.O1_MAX_O2_MAX:
        return np.array([1.0]), np.array([1.0])
    elif mode == pareto2d.Mode.O1_MIN_O2_MIN:
        return np.array([0.0]), np.array([0.0])
    elif mode == pareto2d.Mode.O1_MAX_O2_MIN:
        return np.array([1.0]), np.array([0.0])
    elif mode == pareto2d.Mode.O1_MIN_O2_MAX:
        return np.array([0.0]), np.array([1.0])
    else:
        raise ValueError(f"Unknown {mode=}")


def get_pareto2d_dataset(mode, n, seed=414215135):
    import pareto2d

    if mode == pareto2d.Mode.O1_MAX_O2_MAX:
        o1c = 0.0
        o2c = 0.0
    elif mode == pareto2d.Mode.O1_MIN_O2_MIN:
        o1c = 1.0
        o2c = 1.0
    elif mode == pareto2d.Mode.O1_MAX_O2_MIN:
        o1c = 0.0
        o2c = 1.0
    elif mode == pareto2d.Mode.O1_MIN_O2_MAX:
        o1c = 1.0
        o2c = 0.0

    else:
        raise ValueError(f"Unknown {mode=}")

    rng = np.random.default_rng(seed=3407)

    o1 = rng.random(N)
    o2 = rng.random(N)

    mask = (o1 - o1c) * (o1 - o1c) + (o2 - o2c) * (o2 - o2c) < 1
    return o1[mask], o2[mask]


# %% [markdown]
# # pareto2d
#
# `pareto2d` is a mini library for processing Pareto fronts.
#
# It supports:
#
# * finding Pareto fronts
# * interpolating Pareto fronts
# * flexible configuration related to which objective is maximized and which is minimized
#

# %% [markdown]
# ## Examples intro  - imports and data preparation

# %%
import numpy as np
import matplotlib.pyplot as plt
import pareto2d

# Important - set mode! -> which target is to be minimized and which maximized

mode = pareto2d.Mode.O1_MAX_O2_MAX
# mode = pareto2d.Mode.O1_MAX_O2_MIN
# mode = pareto2d.Mode.O1_MIN_O2_MAX
# mode = pareto2d.Mode.O1_MIN_O2_MIN

# Generate data <- REPLACE THIS SECTION WITH YOUR DATA GENERATION CODE

N = 500
o1, o2 = get_pareto2d_dataset(mode, N)
o1i, o2i = get_pareto2d_ideal_solution(mode)

# %% [markdown]
# ## Example 1 - find Pareto front

# %%
# Find Pareto front
pf_mask = pareto2d.get_pf_mask(o1, o2, mode)
o1pf, o2pf = o1[pf_mask], o2[pf_mask]

# Make plot

fig, ax = plt.subplots(figsize=(6, 6))
cm = plt.get_cmap("tab10")

ax.scatter(x=o1, y=o2, color="lightgray")
ax.scatter(x=o1i, y=o2i, color="red", s=100, label="ideal solution")
ax.scatter(x=o1pf, y=o2pf, color=cm(0), label="Pareto front")

ax.set_xlabel("Objective 1", fontsize=12)
ax.set_ylabel("Objective 2", fontsize=12)
ax.grid()
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])


# %% [markdown]
# ## Example 2 - interpolate Pareto front

# %%
# Find Pareto front
o1pf, o2pf = pareto2d.get_sorted_pf(o1, o2, mode)

# Interpolate second objective from pf on the basis of first objective grid `o1dense`
o1dense = np.arange(0.0, 1.0, 0.001)
o2dense = pareto2d.interpolate_pf_o2(o1dense, o1pf, o2pf, mode)


fig, ax = plt.subplots(figsize=(6, 6))
cm = plt.get_cmap("tab10")

ax.scatter(x=o1, y=o2, color="lightgray")
ax.scatter(x=o1i, y=o2i, color="red", s=100, label="ideal solution")
ax.scatter(x=o1pf, y=o2pf, color=cm(0), label="Pareto front")
ax.plot(o1dense, o2dense, color=cm(0))

ax.set_xlabel("Objective 1", fontsize=12)
ax.set_ylabel("Objective 2", fontsize=12)
ax.grid()
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])


# %%
# Interpolate first objective from Pareto front on the basis of second objective grid `o2dense`
o2dense = np.arange(0.0, 1.0, 0.001)
o1dense = pareto2d.interpolate_pf_o1(o2dense, o1pf, o2pf, mode)

fig, ax = plt.subplots(figsize=(6, 6))
cm = plt.get_cmap("tab10")

ax.scatter(x=o1, y=o2, color="lightgray")
ax.scatter(x=o1i, y=o2i, color="red", s=100, label="ideal solution")
ax.scatter(x=o1pf, y=o2pf, color=cm(0), label="Pareto front")
ax.plot(o1dense, o2dense, color=cm(0))

ax.set_xlabel("Objective 1", fontsize=12)
ax.set_ylabel("Objective 2", fontsize=12)
ax.grid()
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])

# %% [markdown]
# ## Example 3 - find successive Pareto fronts

# %%
# Prepare figure

fig, ax = plt.subplots(figsize=(6, 6))
cm = plt.get_cmap("tab10")

ax.scatter(x=o1, y=o2, color="lightgray")
ax.scatter(x=o1i, y=o2i, color="red", s=100, label="ideal solution")
ax.set_xlabel("Objective 1", fontsize=12)
ax.set_ylabel("Objective 2", fontsize=12)
ax.grid()
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])

# Find points belonging to successive Pareto front

o1_cur = o1
o2_cur = o2

for i in range(0, 5):
    pf_mask = pareto2d.get_pf_mask(o1_cur, o2_cur, mode)
    o1pf_cur, o2pf_cur = o1_cur[pf_mask], o2_cur[pf_mask]
    o1_cur, o2_cur = o1_cur[~pf_mask], o2_cur[~pf_mask]
    ax.scatter(x=o1pf_cur, y=o2pf_cur, color=cm(i), label=f"Pareto front {i+1}")

ax.legend()
