from abc import abstractmethod

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx


class BasicGFNModel(eqx.Module):
    log_z: jax.Array
    side: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, side, dim, log_z_components=64):
        if log_z_components < 1:
            raise ValueError("log_z_components must be positive")
        self.side = side
        self.dim = dim
        self.log_z = jnp.zeros((log_z_components,))

    def get_log_z(self):
        return self.log_z.sum()

    def __call__(self, states):
        # Terminal slots use a valid cell for lookup and are masked in the loss.
        return self.predict(jnp.maximum(states, 0))

    @abstractmethod
    def predict(self, states):
        """Return log flow, forward logits, and backward logits for grid states."""


class TabularGFNModel(BasicGFNModel):
    flow: jax.Array
    forward_policy: jax.Array
    backward_policy: jax.Array

    def __init__(self, side, dim, log_z_components=64):
        super().__init__(side, dim, log_z_components)
        grid_shape = (side,) * dim
        policy_shape = grid_shape + (2 * dim + 1,)
        self.flow = jnp.zeros(grid_shape)
        self.forward_policy = jnp.zeros(policy_shape)
        # Backward logits index the incoming move. The last slot is the source.
        self.backward_policy = jnp.zeros(policy_shape)

    def predict(self, states):
        coordinates = jnp.moveaxis(states, -1, 0)
        idx = tuple(coordinates)
        flow = self.flow[idx]
        forward_logits = self.forward_policy[idx]
        backward_logits = self.backward_policy[idx]
        return flow, forward_logits, backward_logits


class MLPGFNModel(BasicGFNModel):
    network: eqx.nn.MLP

    def __init__(self, side, dim, log_z_components=64, *, key):
        super().__init__(side, dim, log_z_components)
        self.network = eqx.nn.MLP(
            in_size=side * dim, out_size=4 * dim + 3,
            width_size=256, depth=2, key=key,
        )

    def predict(self, states):
        inputs = jax.nn.one_hot(states, self.side).reshape(-1, self.dim * self.side)
        outputs = jax.vmap(self.network)(inputs)
        outputs = outputs.reshape(states.shape[:-1] + (4 * self.dim + 3,))
        num_actions = 2 * self.dim + 1
        return outputs[..., 0], outputs[..., 1:1 + num_actions], outputs[..., 1 + num_actions:]


GFNModel = TabularGFNModel

def valid_actions(state, side):
    return jnp.concatenate((state > 0, state < side - 1, jnp.ones(state.shape[:-1] + (1,), dtype=bool)), axis=-1)

def valid_inverse_actions(state, side):
    return jnp.concatenate((state < side - 1, state > 0, jnp.ones(state.shape[:-1] + (1,), dtype=bool)), axis=-1)

def run_trajectory(model: BasicGFNModel, key, max_length=128):
    dim, side = model.dim, model.side
    stop_action = 2 * dim

    key, start_key = random.split(key)
    start = random.randint(start_key, shape=(dim,), minval=0, maxval=side)

    def trajectory_step(state, key):
        _, forward_logits, _ = model(state)
        masked_logits = jnp.where(valid_actions(state, side), forward_logits, -jnp.inf)
        action = random.categorical(key, masked_logits)

        go_to_end = action == stop_action
        new_state = jax.lax.cond(
            go_to_end,
            lambda: jnp.full_like(state, -1),
            lambda: state.at[action % dim].add(jnp.where(action < dim, -1, 1)),
        )
        return new_state, action

    # Trajectory is padded
    # Trajectory is in fact copied during each step, maybe it's worth optimizing
    trajectory = jnp.full((max_length, dim), -1, dtype=start.dtype).at[0].set(start)
    actions = jnp.full((max_length - 1,), stop_action, dtype=start.dtype)
    if max_length == 1:
        return trajectory, actions

    _, steps_keys_init = random.split(key)
    steps_keys = random.split(steps_keys_init, max_length - 1)

    def has_next_step(carry):
        step, state, _, _ = carry
        return (step < max_length - 1) & (state[0] != -1)

    def while_step(carry):
        step, state, trajectory, actions = carry
        new_state, action = trajectory_step(state, steps_keys[step])
        trajectory = trajectory.at[step + 1].set(new_state)
        actions = actions.at[step].set(action)
        return step + 1, new_state, trajectory, actions

    _, _, trajectory, actions = jax.lax.while_loop(
        has_next_step, while_step, (0, start, trajectory, actions)
    )
    return trajectory, actions

def get_reward(state, side):
    distance = jnp.abs(state / (side - 1) - 0.5)
    outer = jnp.all(distance > 0.25, axis=-1)
    peak = jnp.all((distance > 0.3) & (distance < 0.4), axis=-1)
    return 1e-3 + 0.5 * outer + 2.0 * peak

def calc_loss(model: BasicGFNModel, trajectory, actions):
    dim, side = model.dim, model.side

    length = trajectory.shape[0]
    length_to_done = (trajectory[:, 0] != -1).sum()

    flow, forward_logits, backward_logits = model(trajectory)

    forward_valid = valid_actions(trajectory, side)
    backward_valid = valid_inverse_actions(trajectory, side)

    forward_logits = jnp.where(forward_valid, forward_logits, -jnp.inf)
    backward_logits = jnp.where(backward_valid, backward_logits, -jnp.inf)

    forward_logprobs = jax.nn.log_softmax(forward_logits, axis=1)
    backward_logprobs = jax.nn.log_softmax(backward_logits, axis=1)

    forward_logprobs = jnp.where(forward_valid, forward_logprobs, 0)
    backward_logprobs = jnp.where(backward_valid, backward_logprobs, 0)
    
    # left part of detailed balance 
    rows = jnp.arange(length - 1)
    log_a = forward_logprobs[rows, actions] + flow[:-1]
    # right part of detailed balance
    log_b = backward_logprobs[rows + 1, actions] + flow[1:]

    good_moves = (trajectory[1:, 0] != -1)
    total_loss = jnp.sum(jnp.where(good_moves, log_a - log_b, 0) ** 2)
    total_moves = good_moves.sum()


    # loss for the first move
    log_total_states = jnp.log(side) * dim
    first_log_a = -log_total_states + model.get_log_z()
    first_log_b = backward_logprobs[0, -1] + flow[0]
    total_loss = total_loss + (first_log_a - first_log_b) ** 2
    total_moves = total_moves + 1


    # loss for the last move
    done = length_to_done < length
    finish = trajectory[length_to_done - 1]
    last_log_a = forward_logprobs[length_to_done - 1, -1] + flow[length_to_done - 1]
    last_log_b = jnp.log(get_reward(finish, side))
    total_loss = total_loss + jnp.where(done, (last_log_a - last_log_b) ** 2, 0)
    total_moves = total_moves + done

    # total flow
    good_states = trajectory[:, 0] != -1
    flow_exp = jnp.exp(jnp.where(good_states, flow, 0))
    total_flow = jnp.where(good_states, flow_exp, 0).sum()
    total_flow_states = length_to_done

    return total_loss, total_moves, total_flow, total_flow_states


@eqx.filter_jit
def train_step(
    model, optimizer, opt_state, key, batch_size=64, *,
    max_length=128, return_trajectories=False, flow_regularization_coeff=0.001,
    return_db_loss=False,
):
    key, rollout_key = random.split(key)
    keys = random.split(rollout_key, batch_size)
    # Sample outside the differentiated loss so reverse mode never enters the rollout.
    trajectories, actions = jax.vmap(run_trajectory, in_axes=(None, 0, None))(
        model, keys, max_length
    )

    def batch_loss(model):
        loss_sums, edge_counts, flow_sums, states_counts = jax.vmap(
            calc_loss, in_axes=(None, 0, 0)
        )(model, trajectories, actions)
        basic_loss = loss_sums.sum() / edge_counts.sum()
        flow_regularization = flow_sums.sum() / states_counts.sum()
        return basic_loss + flow_regularization * flow_regularization_coeff, basic_loss

    (loss, db_loss), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    reported_loss = db_loss if return_db_loss else loss
    if return_trajectories:
        return model, opt_state, key, reported_loss, trajectories
    return model, opt_state, key, reported_loss
