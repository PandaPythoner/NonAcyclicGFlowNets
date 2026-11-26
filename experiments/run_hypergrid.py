import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from jax import random
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .hypergrid import MLPGFNModel, TabularGFNModel, get_reward, train_step
else:
    from hypergrid import MLPGFNModel, TabularGFNModel, get_reward, train_step


@eqx.filter_jit
def train_batch(
    model, optimizer, opt_state, key, batch_size, max_length, flow_coeff,
    project_flow_gradient=False,
):
    model, opt_state, key, loss, trajectories = train_step(
        model, optimizer, opt_state, key, batch_size,
        max_length=max_length, return_trajectories=True,
        flow_regularization_coeff=flow_coeff, return_db_loss=True,
        project_flow_gradient=project_flow_gradient,
    )
    # Grid visits include the first cell and exclude the sink and padding.
    lengths = (trajectories[..., 0] != -1).sum(axis=1)
    finished = lengths < trajectories.shape[1]
    last_cells = trajectories[jnp.arange(batch_size), lengths - 1]
    cell_ids = jnp.ravel_multi_index(
        tuple(last_cells.T), (model.side,) * model.dim, mode="clip"
    )
    return model, opt_state, key, loss, lengths.mean(), jnp.where(finished, cell_ids, -1)


@eqx.filter_jit
def train_runs(
    models, optimizer, opt_states, keys, batch_size, max_length, flow_coeffs,
    project_flow_gradient=False,
):
    return eqx.filter_vmap(
        train_batch,
        in_axes=(eqx.if_array(0), None, eqx.if_array(0), 0, None, None, 0, None),
    )(models, optimizer, opt_states, keys, batch_size, max_length, flow_coeffs,
      project_flow_gradient)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def optimizer_labels(params):
    labels = jax.tree.map(lambda _: "model", params)
    return eqx.tree_at(lambda model: model.log_z, labels, "log_z")


def main():
    parser = argparse.ArgumentParser(description="Train a GFlowNet on cyclic Hypergrid.")
    parser.add_argument("--model", choices=("tabular", "mlp"), default="tabular")
    parser.add_argument("--side", type=positive_int, default=7,
                        help="Number of cells along each axis (at least 2).")
    parser.add_argument("--dim", type=positive_int, default=2,
                        help="Number of grid dimensions.")
    parser.add_argument("--steps", type=positive_int, default=10_000)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--window", type=positive_int, default=200_000,
                        help="Number of recent training trajectories in the distribution metric.")
    parser.add_argument("--max-length", type=positive_int, default=128,
                        help="Number of rollout slots, including the first cell.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-z-lr", type=float, default=0.5,
                        help="Adam learning rate for log Z (default: %(default)g).")
    parser.add_argument("--flow-regularization-coeff", type=float, nargs="+", default=[0.001],
                        help="Train one independent run per coefficient in parallel (0 disables the penalty).")
    parser.add_argument("--project-flow-gradient", action="store_true",
                        help="Project the conflicting flow gradient orthogonally to the DB gradient before Adam.")
    parser.add_argument("--output", type=Path, default=Path("hypergrid_metrics.png"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"),
                        help="Root directory for TensorBoard runs.")
    parser.add_argument("--log-every", type=positive_int, default=100,
                        help="Write metrics and update the progress details every N steps.")
    args = parser.parse_args()
    if args.side < 2:
        parser.error("--side must be at least 2 for the reward function")
    if args.max_length < 2:
        parser.error("--max-length must be at least 2 so the policy can choose stop")
    if any(not np.isfinite(coeff) or coeff < 0 for coeff in args.flow_regularization_coeff):
        parser.error("--flow-regularization-coeff must contain finite nonnegative values")
    if not np.isfinite(args.log_z_lr) or args.log_z_lr <= 0:
        parser.error("--log-z-lr must be finite and positive")

    side, dim = args.side, args.dim
    optimizer = optax.partition(
        {"model": optax.adam(args.lr), "log_z": optax.adam(args.log_z_lr)},
        optimizer_labels,
    )
    grid_shape = (side,) * dim
    cells = jnp.indices(grid_shape).reshape(dim, -1).T
    rewards = np.asarray(get_reward(cells, side), dtype=np.float64)
    normalizer = rewards.sum()
    target = rewards / normalizer
    exact_log_z = np.log(normalizer)

    samples = np.arange(1, args.steps + 1) * args.batch_size
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, layout="constrained")
    grid_label = "×".join([str(side)] * dim)
    fig.suptitle(f"Cyclic Hypergrid {grid_label} · {args.model} (seed {args.seed})\n"
                 f"log Z LR: {args.log_z_lr:g} · flow projection: {args.project_flow_gradient}")
    if args.model == "tabular":
        model = TabularGFNModel(side, dim)
    else:
        init_key = random.fold_in(random.key(args.seed), 0)
        model = MLPGFNModel(side, dim, key=init_key)
    histories = run_experiments(args, model, optimizer, target, exact_log_z)
    for coeff, metrics in zip(args.flow_regularization_coeff, histories):
        for ax, values in zip(axes, metrics):
            ax.plot(samples, values, label=f"λ = {coeff:g}")
    for ax, label in zip(axes, ("Batch DB loss", "Endpoint L1 error",
                                "Mean observed length\n(grid visits)", "Learned log Z")):
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend()
    exact_line = axes[-1].axhline(exact_log_z, color="0.25", linestyle="--",
                                 label=f"Exact log Z = {exact_log_z:.5f}")
    axes[-1].legend(handles=[exact_line])
    axes[-1].set_xlabel("Training trajectories sampled")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")


def run_experiments(args, model, optimizer, target, exact_log_z):
    num_runs = len(args.flow_regularization_coeff)

    @eqx.filter_vmap
    def initialize_run(key):
        return model, optimizer.init(eqx.filter(model, eqx.is_inexact_array)), key

    # Match the previous comparison: each coefficient starts from the same seed/model.
    keys = jnp.stack([random.key(args.seed)] * num_runs)
    models, opt_states, keys = initialize_run(keys)
    flow_coeffs = jnp.asarray(args.flow_regularization_coeff)
    recent = [deque() for _ in range(num_runs)]
    counts = np.zeros((num_runs, target.size), dtype=np.int64)
    histories = [[[], [], [], []] for _ in range(num_runs)]

    with ExitStack() as resources:
        writers, progress_bars = [], []
        for run, coeff in enumerate(args.flow_regularization_coeff):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            mode = "projected" if args.project_flow_gradient else "standard"
            run_name = (f"{args.model}_{model.side}pow{model.dim}_lambda{coeff:g}_{mode}_seed{args.seed}_"
                        f"{timestamp}_{uuid4().hex[:8]}")
            log_dir = args.log_dir / run_name
            print(f"TensorBoard run: {log_dir.resolve()}", flush=True)
            config = dict(vars(args), side=model.side, dim=model.dim, flow_regularization_coeff=coeff)
            writer = resources.enter_context(SummaryWriter(str(log_dir), flush_secs=5))
            writer.add_text("config", json.dumps(config, default=str, indent=2), 0)
            writer.add_scalars("log_z/learned", {
                "learned": float(model.get_log_z()), "exact": float(exact_log_z),
            }, 0)
            writer.flush()
            writers.append(writer)
            progress_bars.append(resources.enter_context(tqdm(
                total=args.steps, desc=f"{args.model} λ={coeff:g}", position=run,
                unit="step", dynamic_ncols=True,
            )))

        for step in range(1, args.steps + 1):
            models, opt_states, keys, losses, mean_lengths, batch_cell_ids = train_runs(
                models, optimizer, opt_states, keys, args.batch_size, args.max_length, flow_coeffs,
                args.project_flow_gradient,
            )
            losses, mean_lengths, batch_cell_ids, log_z_values = jax.device_get(
                (losses, mean_lengths, batch_cell_ids, models.get_log_z())
            )
            for run, (writer, progress) in enumerate(zip(writers, progress_bars)):
                loss, mean_length, log_z = losses[run], mean_lengths[run], log_z_values[run]
                for cell_id in batch_cell_ids[run].tolist():
                    if len(recent[run]) == args.window:
                        oldest = recent[run].popleft()
                        if oldest >= 0:
                            counts[run, oldest] -= 1
                    recent[run].append(cell_id)
                    if cell_id >= 0:
                        counts[run, cell_id] += 1

                completed = counts[run].sum()
                distance = np.abs(counts[run] / completed - target).sum() if completed else np.nan
                for history, value in zip(histories[run], (loss, distance, mean_length, log_z)):
                    history.append(float(value))

                if step == 1 or step == args.steps or step % args.log_every == 0:
                    truncated = 1 - completed / len(recent[run])
                    metrics = {
                        "loss/db": float(loss), "sampling/endpoint_l1": float(distance),
                        "sampling/mean_observed_length": float(mean_length),
                        "sampling/truncated_fraction": float(truncated),
                        "log_z/absolute_error": float(abs(log_z - exact_log_z)),
                    }
                    samples = step * args.batch_size
                    for name, value in metrics.items():
                        writer.add_scalar(name, value, samples)
                    writer.add_scalars("log_z/learned", {
                        "learned": float(log_z), "exact": float(exact_log_z),
                    }, samples)
                    progress.set_postfix(
                        DB=f"{loss:.3g}", L1=f"{distance:.4f}", length=f"{mean_length:.1f}",
                        log_Z=f"{log_z:.4f}", truncated=f"{truncated:.1%}", refresh=False,
                    )
                progress.update(1)

    return histories


if __name__ == "__main__":
    main()
