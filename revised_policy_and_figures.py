"""Validation-selected, feedback-aware policy evaluation and figures.

This script is intended to run after notebook 02 has been updated to save,
for every fold, validation and test predictions together with
``*_observed_power_db``.  It deliberately separates validation-time
parameter selection from held-out test reporting.

Required prediction-file keys for each split (``val`` and ``test``):
    {split}_target_db, {split}_Yv, {split}_observed_power_db,
    {split}_best, and ``{model}_mean_db``/``{model}_std_db``/
    ``{model}_link_prob`` for each available model.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(os.environ.get("WCNC_PROJECT_ROOT", Path.home() / "wcnc_project"))
RESULTS_DIR = PROJECT_ROOT / "improved_experiment_v2" / "results"
FIGURE_DIR = PROJECT_ROOT / "improved_experiment_v2" / "figures_revised"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["persistence", "deterministic", "independent", "joint", "hurdle_joint"]
SHORT = {
    "persistence": "Persistence",
    "deterministic": "Deterministic GRU",
    "independent": "Independent probabilistic GRU",
    "joint": "Joint probabilistic GRU",
    "hurdle_joint": "Hurdle-aware joint GRU",
}
COLORS = {
    "persistence": "#7f7f7f",
    "deterministic": "#009e73",
    "independent": "#e69f00",
    "joint": "#d55e00",
    "hurdle_joint": "#0072b2",
}

FLOOR_DB = -220.0
POLICIES = ("legacy_margin", "dp_ttt")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def dp_first_action(step_scores: np.ndarray, current: int, cost_db: float) -> int:
    """Backward-induction first action for one receding-horizon window."""
    horizon, n_gnb = step_scores.shape
    value_next = np.zeros(n_gnb, dtype=float)
    first_policy = None
    for h in range(horizon - 1, -1, -1):
        value_now = np.full(n_gnb, -np.inf, dtype=float)
        action_now = np.zeros(n_gnb, dtype=np.int16)
        for previous in range(n_gnb):
            values = np.array([
                step_scores[h, candidate]
                - (cost_db if candidate != previous else 0.0)
                + value_next[candidate]
                for candidate in range(n_gnb)
            ])
            action_now[previous] = int(np.argmax(values))
            value_now[previous] = values[action_now[previous]]
        if h == 0:
            first_policy = action_now
        value_next = value_now
    return int(first_policy[current])


def risk_aware_scores(mean_db, std_db, link_prob, beta, failure_penalty_db,
                      reliability):
    """Convert forecast mean/uncertainty/availability to a decision score.

    The lower confidence term makes uncertainty affect the decision directly.
    The availability term uses a conservative expected-power mixture, and the
    failure penalty makes the probability of an unavailable link explicit.
    ``reliability`` is updated only from previous execution outcomes.
    """
    mean_db = np.asarray(mean_db, dtype=float)
    if std_db is None:
        std_db = np.zeros_like(mean_db)
    if link_prob is None:
        link_prob = np.ones_like(mean_db)
    else:
        link_prob = np.asarray(link_prob, dtype=float)

    p = np.clip(link_prob * reliability[None, None, :], 1e-4, 1.0)
    conservative_db = mean_db - beta * np.asarray(std_db, dtype=float)
    usable_linear = np.power(10.0, np.clip(conservative_db, FLOOR_DB, 0.0) / 10.0)
    floor_linear = 10.0 ** (FLOOR_DB / 10.0)
    expected_linear = p * usable_linear + (1.0 - p) * floor_linear
    score_db = 10.0 * np.log10(np.maximum(expected_linear, floor_linear))
    score_db -= failure_penalty_db * (1.0 - p)
    return score_db


def run_feedback_dp_ttt(mean_db, std_db, link_prob, observed_power_db,
                        first_step_valid, beta, cost_db, failure_penalty_db,
                        ttt_k, feedback_decay=0.8):
    """Run receding-horizon DP with TTT and online execution reliability.

    ``first_step_valid[i, action]`` is used only after action i is selected,
    representing feedback available before the next decision epoch.
    """
    n_windows, _, n_gnb = mean_db.shape
    actions = np.zeros(n_windows, dtype=np.int16)
    recommendations = np.zeros(n_windows, dtype=np.int16)
    serving = int(np.argmax(observed_power_db[0]))
    reliability = np.ones(n_gnb, dtype=float)
    pending, streak = None, 0

    for i in range(n_windows):
        p_i = None if link_prob is None else link_prob[i:i + 1]
        s_i = None if std_db is None else std_db[i:i + 1]
        scores = risk_aware_scores(mean_db[i:i + 1], s_i, p_i, beta,
                                   failure_penalty_db, reliability)[0]
        recommendation = dp_first_action(scores, serving, cost_db)
        recommendations[i] = recommendation

        if recommendation == serving:
            pending, streak = None, 0
        else:
            if recommendation == pending:
                streak += 1
            else:
                pending, streak = recommendation, 1
            if streak >= ttt_k:
                serving = recommendation
                pending, streak = None, 0

        actions[i] = serving

        # Execution feedback is applied only after the current action.
        success = float(first_step_valid[i, serving] > 0)
        reliability[serving] = (
            feedback_decay * reliability[serving]
            + (1.0 - feedback_decay) * success
        )

    return actions, recommendations


def run_legacy_policy(score_db, observed_power_db, margin_db=2.0):
    raw = np.argmax(score_db.mean(axis=1), axis=1).astype(np.int16)
    actions = np.zeros(len(raw), dtype=np.int16)
    serving = int(np.argmax(observed_power_db[0]))
    for i, candidate in enumerate(raw):
        if candidate != serving:
            gain = score_db[i].mean(axis=0)[candidate] - score_db[i].mean(axis=0)[serving]
            if gain >= margin_db:
                serving = int(candidate)
        actions[i] = serving
    return actions


def evaluate(actions, target_db, valid):
    n, horizon, _ = target_db.shape
    rows = np.arange(n)[:, None]
    steps = np.arange(horizon)[None, :]
    selected = target_db[rows, steps, actions[:, None]]
    selected_valid = valid[rows, steps, actions[:, None]].astype(bool)
    usable = np.where(valid.astype(bool), target_db, -np.inf)
    best_usable = np.max(usable, axis=2)
    oracle = np.max(target_db, axis=2)

    valid_selected = selected_valid & np.isfinite(best_usable)
    valid_gap = (
        float(np.mean(best_usable[valid_selected] - selected[valid_selected]))
        if valid_selected.any() else np.nan
    )
    selected_valid_power = (
        float(np.mean(selected[valid_selected])) if valid_selected.any() else np.nan
    )
    return {
        "handover_count": int(np.sum(np.diff(actions) != 0)),
        "invalid_action_rate": float(1.0 - np.mean(selected_valid[:, 0])),
        "sustained_invalid_rate": float(1.0 - np.mean(np.all(selected_valid, axis=1))),
        "first_step_validity": float(np.mean(selected_valid[:, 0])),
        "valid_link_gap_db": valid_gap,
        "selected_valid_power_db": selected_valid_power,
        "raw_oracle_gap_db": float(np.mean(oracle - selected)),
    }


def selection_key(metrics):
    """Safety-first validation selection, without using held-out test data."""
    invalid = metrics["invalid_action_rate"]
    valid_gap = metrics["valid_link_gap_db"]
    switches = metrics["handover_count"]
    return (invalid, valid_gap, switches)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def add_paired_panel(ax, report, metric, xlabel, title, scale=1.0):
    """Plot legacy-to-DP paired changes without hiding model dependence."""
    y = np.arange(len(MODELS))[::-1]
    for model, yi in zip(MODELS, y):
        legacy = np.array([r[metric] for r in report["test_metrics"][model]["legacy_margin"]]) * scale
        proposed = np.array([r[metric] for r in report["test_metrics"][model]["dp_ttt"]]) * scale
        ml, sl = mean_std(legacy)
        mp, sp = mean_std(proposed)
        ax.plot([ml, mp], [yi, yi], color=COLORS[model], lw=1.8, alpha=0.7,
                solid_capstyle="round", zorder=1)
        ax.errorbar([ml, mp], [yi, yi], xerr=[[sl, sp], [sl, sp]], fmt="none",
                    ecolor=COLORS[model], elinewidth=0.7, capsize=1.5, alpha=0.8,
                    zorder=2)
        ax.scatter(ml, yi, s=30, facecolor="white", edgecolor=COLORS[model],
                   linewidth=1.2, zorder=3)
        ax.scatter(mp, yi, s=30, facecolor=COLORS[model], edgecolor="black",
                   linewidth=0.5, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in MODELS])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(left=0)


def save_publication_figures(report, forecast_summary):
    """Create figures that explicitly connect forecasting to decision quality."""
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.25),
                             gridspec_kw={"wspace": 0.24, "hspace": 0.42})

    # (a) Forecast accuracy, with every held-out trajectory visible.
    ax = axes[0, 0]
    y = np.arange(len(MODELS))[::-1]
    for model, yi in zip(MODELS, y):
        values = np.array([
            forecast_summary["per_fold_results"][str(k)][model]["masked_mae_db"]
            for k in range(int(forecast_summary["num_folds"]))
        ])
        mu, sd = mean_std(values)
        ax.hlines(yi, 0, mu, color=COLORS[model], lw=1.2, alpha=0.55)
        ax.scatter(values, yi + rng.uniform(-0.11, 0.11, len(values)),
                   color=COLORS[model], s=10, alpha=0.55, linewidths=0)
        ax.errorbar(mu, yi, xerr=sd, fmt="D", ms=5, color=COLORS[model],
                    mec="black", mew=0.45, ecolor="black", elinewidth=0.8,
                    capsize=2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in MODELS])
    ax.set_xlabel("Masked path-power MAE (dB)")
    ax.set_title("(a) Forecast accuracy", loc="left", fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(left=0)

    # (b) Switching behavior.
    add_paired_panel(axes[0, 1], report, "handover_count",
                     "Target-cell switches per trajectory",
                     "(b) Policy switching")

    # (c) Execution validity, which is more informative than raw oracle gap
    # when invalid links are represented by a sentinel power floor.
    add_paired_panel(axes[1, 0], report, "invalid_action_rate",
                     "Invalid first-step actions (%)",
                     "(c) Execution validity", scale=100.0)

    # (d) Decision quality among usable targets.
    add_paired_panel(axes[1, 1], report, "valid_link_gap_db",
                     "Gap to best usable link (dB)",
                     "(d) Usable-target alignment")

    legend = [
        Line2D([0], [0], marker="o", ls="", mfc="white", mec="black",
               mew=1.0, ms=5, label="Legacy margin"),
        Line2D([0], [0], marker="o", ls="", mfc="black", mec="black",
               ms=5, label="Validation-selected DP--TTT"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.59, -0.015), columnspacing=1.6,
               handletextpad=0.3)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.96, bottom=0.13)
    fig.savefig(FIGURE_DIR / "fig_forecast_to_policy.pdf")
    fig.savefig(FIGURE_DIR / "fig_forecast_to_policy.png")
    plt.close(fig)

    # The raw future-power oracle gap is retained as a supplementary diagnostic.
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    add_paired_panel(ax, report, "raw_oracle_gap_db", "Raw oracle gap (dB)",
                     "Supplementary: unconstrained oracle gap")
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, fontsize=6.5)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.90, bottom=0.25)
    fig.savefig(FIGURE_DIR / "fig_raw_oracle_gap_supplement.pdf")
    plt.close(fig)


def parameter_grid():
    return itertools.product(
        [0.0, 0.25, 0.5, 1.0],       # uncertainty coefficient beta
        [0.5, 1.0, 2.0, 4.0],         # DP handover cost in dB
        [0.0, 2.0, 5.0, 10.0],        # failure penalty in dB
        [1, 2, 3],                    # TTT decision epochs
        [0.6, 0.8, 0.95],             # feedback-memory decay
    )


def require_key(data, key):
    if key not in data:
        raise KeyError(
            f"Missing '{key}'. Rerun notebook 02 after saving validation "
            "predictions and the last observed power for each fold."
        )
    return data[key]


def prediction_key(data, split, model, suffix):
    """Read new split-prefixed keys, with backward compatibility for test keys."""
    preferred = f"{split}_{model}_{suffix}"
    if preferred in data:
        return data[preferred]
    if split == "test":
        legacy = f"{model}_{suffix}"
        if legacy in data:
            return data[legacy]
    raise KeyError(f"Missing prediction key '{preferred}'.")


def main():
    summary_file = RESULTS_DIR / "forecaster_loto_summary.json"
    with open(summary_file, encoding="utf-8") as f:
        forecast_summary = json.load(f)

    n_folds = int(forecast_summary["num_folds"])
    selected_params = {}
    selected_validation_metrics = {}
    test_metrics = {m: {p: [] for p in POLICIES} for m in MODELS}

    for fold in range(n_folds):
        path = RESULTS_DIR / f"fold_{fold}_summary_predictions.npz"
        with np.load(path, allow_pickle=True) as d:
            val_target = require_key(d, "val_target_db")
            val_valid = require_key(d, "val_Yv")
            val_observed = require_key(d, "val_observed_power_db")
            test_target = require_key(d, "test_target_db")
            test_valid = require_key(d, "test_Yv")
            test_observed = require_key(d, "test_observed_power_db")

            for model in MODELS:
                val_mean = prediction_key(d, "val", model, "mean_db")
                test_mean = prediction_key(d, "test", model, "mean_db")
                val_std = (prediction_key(d, "val", model, "std_db")
                           if f"val_{model}_std_db" in d else None)
                test_std = (prediction_key(d, "test", model, "std_db")
                            if f"test_{model}_std_db" in d or f"{model}_std_db" in d else None)
                val_prob = (prediction_key(d, "val", model, "link_prob")
                            if f"val_{model}_link_prob" in d else None)
                test_prob = (prediction_key(d, "test", model, "link_prob")
                             if f"test_{model}_link_prob" in d or f"{model}_link_prob" in d else None)

                best = None
                for beta, cost, failure, ttt, decay in parameter_grid():
                    val_actions, _ = run_feedback_dp_ttt(
                        val_mean, val_std, val_prob, val_observed,
                        val_valid[:, 0, :], beta, cost, failure, ttt, decay
                    )
                    candidate = evaluate(val_actions, val_target, val_valid)
                    if best is None or selection_key(candidate) < selection_key(best["metrics"]):
                        best = {
                            "params": {
                                "beta": beta, "handover_cost_db": cost,
                                "failure_penalty_db": failure, "ttt_k": ttt,
                                "feedback_decay": decay,
                            },
                            "metrics": candidate,
                        }

                selected_params.setdefault(model, []).append(best["params"])
                selected_validation_metrics.setdefault(model, []).append(best["metrics"])

                test_actions, _ = run_feedback_dp_ttt(
                    test_mean, test_std, test_prob, test_observed,
                    test_valid[:, 0, :],
                    beta=best["params"]["beta"],
                    cost_db=best["params"]["handover_cost_db"],
                    failure_penalty_db=best["params"]["failure_penalty_db"],
                    ttt_k=best["params"]["ttt_k"],
                    feedback_decay=best["params"]["feedback_decay"],
                )
                test_metrics[model]["dp_ttt"].append(
                    evaluate(test_actions, test_target, test_valid)
                )

                # The legacy policy is retained as a transparent reference.
                legacy_score = risk_aware_scores(
                    test_mean, test_std, test_prob, beta=0.0,
                    failure_penalty_db=0.0,
                    reliability=np.ones(test_mean.shape[-1])
                )
                legacy_actions = run_legacy_policy(legacy_score, test_observed, margin_db=2.0)
                test_metrics[model]["legacy_margin"].append(
                    evaluate(legacy_actions, test_target, test_valid)
                )

    def aggregate(model, policy, metric):
        values = np.array([r[metric] for r in test_metrics[model][policy]], dtype=float)
        return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))

    report = {
        "selected_parameters_by_fold": selected_params,
        "selected_validation_metrics_by_fold": selected_validation_metrics,
        "test_metrics": test_metrics,
        "aggregates": {
            m: {
                p: {
                    metric: aggregate(m, p, metric)
                    for metric in test_metrics[m][p][0]
                }
                for p in POLICIES
            }
            for m in MODELS
        },
    }
    with open(RESULTS_DIR / "revised_policy_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_publication_figures(report, forecast_summary)
    print("Saved validation-selected policy report to:", RESULTS_DIR / "revised_policy_report.json")
    print("Saved publication figures to:", FIGURE_DIR)


if __name__ == "__main__":
    main()
