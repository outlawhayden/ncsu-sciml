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
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

seed = 42
np.random.seed(seed)
key = jax.random.key(seed)
depth = 2
width = 50


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

print("loading data...")
try:
    data = np.load('/Users/haydenoutlaw/Documents/Courses/SciML/ncsu-sciml/hw2/hw2_p3_data.npz')
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

# model architecture, initialize model
# activation function specified here. deserializes from 'dnn_model_opt.eqx'
arch = [x_tr.shape[1]] + [width] * depth + [y_tr.shape[1]]
new_model = MLP(arch, key, activation=jax.nn.gelu)
new_model = eqx.tree_deserialise_leaves("dnn_model_opt.eqx", new_model)


# sample random points from x_tr
key, subkey = jax.random.split(key)
num_samples = 500
idx = jax.random.choice(subkey, x_tr.shape[0], (num_samples,), replace=False)

X_sample = jnp.array(x_tr[idx])
Y_true = jnp.array(y_tr[idx]).flatten()
Y_pred = jax.vmap(lambda x: new_model(x), in_axes=0)(X_sample)

# convert everything to flat numpy arrays of shape (500,)
X_sample = np.array(X_sample)
Y_true   = np.array(Y_true).reshape(-1)
Y_pred   = np.array(Y_pred).reshape(-1)
print(Y_true.shape)
print(Y_pred.shape)
errors   = np.abs(Y_true - Y_pred)

from scipy.interpolate import griddata

# define regular grid on domain
n_grid = 1000
x1_min, x1_max = x_tr[:,0].min(), x_tr[:,0].max()
x2_min, x2_max = x_tr[:,1].min(), x_tr[:,1].max()
xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, n_grid),
    np.linspace(x2_min, x2_max, n_grid)
)

# flatten grid for prediction
X_grid = jnp.array(np.column_stack([xx1.ravel(), xx2.ravel()]))

# predict on grid via vmap
Y_grid_pred = jax.vmap(new_model)(X_grid)
Y_grid_pred = np.array(Y_grid_pred).reshape(n_grid, n_grid)

# interpolate y_tr to new domain
Y_grid_true = griddata(x_tr, y_tr.flatten(), (xx1, xx2), method='cubic')

# compute max/min for colorbar
vmin = min(np.nanmin(Y_grid_true), np.nanmin(Y_grid_pred))
vmax = max(np.nanmax(Y_grid_true), np.nanmax(Y_grid_pred))

# render heatmaps of soln surface
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(
    Y_grid_true,
    extent=(x1_min, x1_max, x2_min, x2_max),
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
axes[0].set_title("True Surface (interpolated)")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

im2 = axes[1].imshow(
    Y_grid_pred,
    extent=(x1_min, x1_max, x2_min, x2_max),
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
axes[1].set_title("Predicted Surface")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
plt.tight_layout()

fig.colorbar(im1, ax=axes, orientation="vertical", shrink=0.7, location="right")


plt.savefig("sample_pred.png", dpi=300)
plt.show()
