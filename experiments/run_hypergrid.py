import argparse
from collections import deque
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from jax import random

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .hypergrid import GFNModel, get_reward, train_step
else:
    from hypergrid import GFNModel, get_reward, train_step


@eqx.filter_jit
def train_batch(model, optimizer, opt_state, key, batch_size, max_length, flow_coeff):
    model, opt_state, key, loss, trajectories = train_step(
        model, optimizer, opt_state, key, batch_size,
        max_length=max_length, return_trajectories=True,
        flow_regularization_coeff=flow_coeff, return_db_loss=True,
    )
    # Grid visits include the first cell and exclude the sink and padding.
    lengths = (trajectories[..., 0] != -1).sum(axis=1)
    finished = lengths < trajectories.shape[1]
    last_cells = trajectories[jnp.arange(batch_size), lengths - 1]
    cell_ids = jnp.ravel_multi_index(
        tuple(last_cells.T), (model.side,) * model.dim, mode="clip"
    )
    return model, opt_state, key, loss, lengths.mean(), jnp.where(finished, cell_ids, -1)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description="Train a tabular GFlowNet on cyclic Hypergrid.")
    parser.add_argument("--steps", type=positive_int, default=10_000)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--window", type=positive_int, default=200_000,
                        help="Number of recent training trajectories in the distribution metric.")
    parser.add_argument("--max-length", type=positive_int, default=128,
                        help="Number of rollout slots, including the first cell.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--flow-regularization-coeff", type=float, nargs="+", default=[0.001],
                        help="One or more flow penalty coefficients to compare (0 disables it).")
    parser.add_argument("--output", type=Path, default=Path("hypergrid_metrics.png"))
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error("--max-length must be at least 2 so the policy can choose stop")
    if any(not np.isfinite(coeff) or coeff < 0 for coeff in args.flow_regularization_coeff):
        parser.error("--flow-regularization-coeff must contain finite nonnegative values")

    side, dim = 7, 2
    optimizer = optax.adam(args.lr)
    grid_shape = (side,) * dim
    cells = jnp.indices(grid_shape).reshape(dim, -1).T
    rewards = np.asarray(get_reward(cells, side), dtype=np.float64)
    normalizer = rewards.sum()
    target = rewards / normalizer
    exact_log_z = np.log(normalizer)

    samples = np.arange(1, args.steps + 1) * args.batch_size
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, layout="constrained")
    fig.suptitle(f"Cyclic Hypergrid {side}×{side} (seed {args.seed})")
    for coeff in args.flow_regularization_coeff:
        model = GFNModel(side, dim)
        metrics = run_experiment(args, model, optimizer, target, coeff)
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


def run_experiment(args, model, optimizer, target, coeff):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    key = random.key(args.seed)
    flow_coeff = jnp.asarray(coeff)
    recent = deque()
    counts = np.zeros(target.size, dtype=np.int64)
    losses, distances, mean_lengths, log_z_values = [], [], [], []

    for step in range(1, args.steps + 1):
        model, opt_state, key, loss, mean_length, cell_ids = train_batch(
            model, optimizer, opt_state, key, args.batch_size, args.max_length, flow_coeff
        )
        loss, mean_length, cell_ids, log_z = jax.device_get(
            (loss, mean_length, cell_ids, model.get_log_z())
        )
        for cell_id in cell_ids.tolist():
            if len(recent) == args.window:
                oldest = recent.popleft()
                if oldest >= 0:
                    counts[oldest] -= 1
            recent.append(cell_id)
            if cell_id >= 0:
                counts[cell_id] += 1

        completed = counts.sum()
        distance = np.abs(counts / completed - target).sum() if completed else np.nan
        losses.append(float(loss))
        distances.append(distance)
        mean_lengths.append(float(mean_length))
        log_z_values.append(float(log_z))

        if step == 1 or step == args.steps or step % max(1, args.steps // 10) == 0:
            truncated = 1 - completed / len(recent)
            print(
                f"λ={coeff:g}  {step}/{args.steps}  DB={loss:.5g}  L1={distance:.4f}  "
                f"mean length={mean_length:.2f}  log Z={log_z:.5f}  "
                f"truncated in window={truncated:.1%}",
                flush=True,
            )

    return losses, distances, mean_lengths, log_z_values


if __name__ == "__main__":
    main()
