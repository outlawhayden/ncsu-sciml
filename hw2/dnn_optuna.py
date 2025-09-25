import jax.numpy as jnp
import numpy as np
import jax
import platform
from flax import nnx
import optax
import equinox as eqx
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import optuna
import time

## BACKEND AUTOCONFIG
print("\nconfiguring backend...")
system = platform.system()
machine = platform.machine().lower()

if system == "Darwin" and ("arm" in machine or "apple" in machine or "m1" in machine or "m2" in machine):
    try:
        jax.config.update("jax_platform_name", "METAL")
        print("Configured JAX backend: metal (Apple Silicon)")
    except Exception as e:
        print("Metal not available, falling back to default:", e)
elif system == "Linux":
    devices = jax.devices()
    if any(d.platform == "gpu" for d in devices):
        jax.config.update("jax_platform_name", "gpu")
        print("Configured JAX backend: gpu")
    else:
        jax.config.update("jax_platform_name", "cpu")
        print("Configured JAX backend: cpu")
else:
    jax.config.update("jax_platform_name", "cpu")
    print("Configured JAX backend: cpu")

print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")

# Seed
seed = 42
np.random.seed(seed)
key = jax.random.key(seed)

num_epochs = 10000   # maximum epochs
width = 50           # fixed hidden width
patience = 200       # early stopping patience

# Load dataset
print("loading data...")
try:
    data = np.load('hw2_p3_data.npz')
    x_tr = data['x_tr']
    y_tr = data['y_tr']
    print("dataset loaded.\n")
except:
    raise RuntimeError("error loading dataset.\n")

indices = np.arange(len(x_tr))
train_idx, test_idx = train_test_split(indices, test_size=0.33, random_state=seed)

X_train, X_test = jnp.array(x_tr[train_idx]), jnp.array(x_tr[test_idx])
y_train, y_test = jnp.array(y_tr[train_idx]), jnp.array(y_tr[test_idx])

print("dataset format:")
print("xtr:", X_train.shape, "xts:", X_test.shape)
print("ytr:", y_train.shape, "yts", y_test.shape)


# === Primitive definitions ===
class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size))

    def __call__(self, x):
        return self.weight @ x + self.bias


class MLP(eqx.Module):
    layers: list
    activations: list

    def __init__(self, architecture, key, activation=jax.nn.relu):
        keys = jax.random.split(key, len(architecture) - 1)
        self.layers = [
            Linear(architecture[i], architecture[i+1], keys[i])
            for i in range(len(architecture) - 1)
        ]
        self.activations = [activation] * (len(self.layers) - 1) + [eqx.nn.Identity()]

    def __call__(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x


# === Loss ===
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

@eqx.filter_jit
def train_step(model, opt_state, x, y, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

@eqx.filter_jit
def eval_step(model, x, y):
    return loss_fn(model, x, y)


# === Activations to sweep ===
acts = {
    "sigmoid": jax.nn.sigmoid,
    "relu": jax.nn.relu,
    "relu6": jax.nn.relu6,
    "gelu": jax.nn.gelu,
    "softmax": jax.nn.softmax,
    "softplus": jax.nn.softplus,
    "leaky_relu": jax.nn.leaky_relu,
}

# === Trial timings and epochs storage ===
trial_times = []
trial_epochs = []


# === Optuna objective ===
def objective(trial):
    start_time = time.time()

    act_name = trial.suggest_categorical("activation", list(acts.keys()))
    depth = trial.suggest_int("depth", 2, 5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)

    activation = acts[act_name]
    arch = [x_tr.shape[1]] + [width] * depth + [y_tr.shape[0]]
    model = MLP(arch, key, activation=activation)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_hist, test_hist = [], []
    best_loss = float("inf")
    patience_counter = 0
    epochs_run = 0

    for epoch in range(num_epochs):
        model, opt_state, train_loss = train_step(model, opt_state, X_train, y_train, optimizer)
        test_loss = eval_step(model, X_test, y_test)

        train_hist.append(float(train_loss))
        test_hist.append(float(test_loss))
        epochs_run = epoch + 1

        # Early stopping
        if test_loss < best_loss - 1e-8:
            best_loss = float(test_loss)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Trial {trial.number}: Early stopping at epoch {epoch}")
                break

    # Save trajectories and metadata
    np.savez(
        f"trial_{trial.number}_losses.npz",
        train=np.array(train_hist),
        test=np.array(test_hist),
        act=act_name,
        depth=depth,
        lr=lr,
        epochs_run=epochs_run,
    )

    # Record time and epoch count
    elapsed = time.time() - start_time
    trial_times.append(elapsed)
    trial_epochs.append(epochs_run)
    print(f"Trial {trial.number} runtime: {elapsed:.2f} seconds, epochs run: {epochs_run}")

    return best_loss


# === Run study ===
if __name__ == "__main__":
    overall_start = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # adjust n_trials as desired
    total_elapsed = time.time() - overall_start

    # Save timings and epochs
    np.savez(
        "timings.npz",
        trial_times=np.array(trial_times),
        trial_epochs=np.array(trial_epochs),
        total_time=total_elapsed,
    )

        # Save best parameters
    trial = study.best_trial
    best_params = trial.params
    np.savez(
        "best_params.npz",
        **best_params,
        best_value=trial.value,
        total_time=total_elapsed,
    )


    print("Best trial:")
    trial = study.best_trial
    print("  Value:", trial.value)
    print("  Params:", trial.params)
    print(f"Total study runtime: {total_elapsed:.2f} seconds")
