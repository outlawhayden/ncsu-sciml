import jax.numpy as jnp
import numpy as np
import jax
import platform
from flax import nnx
import optax
import scipy
from scipy.spatial.distance import cdist
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import equinox as eqx
from tqdm import tqdm

## BACKEND AUTOCONFIG - find GPU if it's there
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


seed = 42
np.random.seed(seed)
key = jax.random.key(seed)
lr = 1e-3
num_epochs = 100000
width = 50  # fixed hidden width


# load dataset, t/t split
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


# primitive definitions
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


# loss fn (L2)
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

# train and eval step
@eqx.filter_jit
def train_step(model, opt_state, x, y, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

@eqx.filter_jit
def eval_step(model, x, y):
    return loss_fn(model, x, y)

# iterate over depths
all_train_hist = {}
all_test_hist = {}

for depth in range(2, 6):  # depths 2 → 5
    print(f"\n=== Training model with depth {depth} ===")
    # Build architecture: input, [hidden]*depth, output
    arch = [x_tr.shape[1]] + [width] * depth + [y_tr.shape[1]]
    model = MLP(arch, key, activation=jax.nn.relu)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_hist, test_hist = [], []
    for epoch in tqdm(range(num_epochs)):
        model, opt_state, train_loss = train_step(model, opt_state, X_train, y_train, optimizer)
        test_loss = eval_step(model, X_test, y_test)

        train_hist.append(float(train_loss))
        test_hist.append(float(test_loss))

    all_train_hist[f"depth{depth}"] = np.array(train_hist)
    all_test_hist[f"depth{depth}"] = np.array(test_hist)

    print(f"Final test loss (depth {depth}):", test_hist[-1])


print("saving out results...")
np.savez("dnn_losses_depth_sweep.npz", **all_train_hist, **{k+"_test": v for k,v in all_test_hist.items()})
print("loss histories saved to dnn_losses_depth_sweep.npz")
