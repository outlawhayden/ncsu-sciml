import jax
from jax import random
from jax.nn import tanh
from jax import numpy as jnp
from jax import vmap, value_and_grad, jit
import matplotlib.pyplot as plt
import numpy as np


jax.config.update("jax_platform_name", "cpu")



def get_random_layer_params(m, n, random_key, init_scheme, init_type):
    w_key, b_key = random.split(random_key)
    if init_type == 'uniform':
        if init_scheme == 'LeCun':
            factor = jnp.sqrt(3/n)
        if init_scheme == 'Xavier':
            factor = jnp.sqrt(6/(m+n))
        if init_scheme == 'He':
            factor = jnp.sqrt(6/n)
        weights = factor * random.uniform(w_key, (n, m), minval=-1.0, maxval=1.0) # Xavier Uniform
           
    if init_type == 'normal':
        if init_scheme == 'LeCun':
            factor = jnp.sqrt(1/n)
        if init_scheme == 'Xavier':
            factor = jnp.sqrt(2/(m+n))
        if init_scheme == 'He':
            factor = jnp.sqrt(2/n)
        weights = factor * random.normal(w_key, (n, m)) # Xavier Normal 
    
    biases  = jnp.zeros((n,)) 
    return weights, biases

def get_init_network_params(sizes, ran_key, init_scheme, init_type):
    keys = random.split(ran_key, len(sizes))
    return [get_random_layer_params(m, n, k, init_scheme, init_type) \
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def feedforward_NN(params, x):
    for w, b in params[:-1]:
        outputs = jnp.dot(w, x) + b  
        x = tanh(outputs)
    w_final, b_final = params[-1] 
    final_outputs = jnp.dot(w_final, x) + b_final 
    return final_outputs


SEED = 1234
key = random.PRNGKey(SEED)

init_scheme  = 'Xavier' # Xavier, LeCun, He
init_type    = 'normal' # normal, uniform
dim_input    = 1
dim_output   = 1
depth        = 2
width        = 32 
layers       = [dim_input] + [width]*depth + [dim_output]
print(layers)
key, params_init_key = random.split(key)
params = get_init_network_params(layers, params_init_key, init_scheme, init_type)


# Training Data Generation

# Define a target function
def f(x):
    return x**2

# Set the number of training data
num_tr_data = 1000

# Generate training data
x_tr = jnp.linspace(-1, 1, num_tr_data)
x_tr = x_tr.reshape((num_tr_data, dim_input)) # Shape: (# data, input dimension)
y_tr = f(x_tr) # Shape: (# data, output dimension)
print(f'input data shape {x_tr.shape}, output data shape {y_tr.shape}')

batched_prediction = vmap(feedforward_NN, in_axes=(None, 0))

@jit
def mse_loss(params, x, y):
    preds = batched_prediction(params, x)
    diff = preds - y
    return jnp.sum(diff*diff)/preds.shape[0]

# standard gradient descent method
@jit
def update(params, x, y, learning_rate):
    l, grads = value_and_grad(mse_loss)(params, x, y)
    return [(w - learning_rate * dw, b - learning_rate * db) 
            for (w, b), (dw, db) in zip(params, grads)], l

# set a learning rate
lr = 1e-2
# set the max number of iterations
maxITER = 100_000

import time
start_time = time.time()
log_loss = []
for it in range(maxITER):
    params, loss = update(params, x_tr, y_tr, lr)
    log_loss.append(loss)
    if (it % 10000 == 0) or (it == maxITER-1):
        end_time = time.time()
        print(f"iter={it:d}, loss={loss:.2e}, lr={lr:.2e}, time took: {end_time-start_time:.2f}s")
        start_time = time.time()
log_loss.append(mse_loss(params, x_tr, y_tr))


plt.figure()
plt.semilogy(np.array(log_loss))
plt.xlabel('the number of iterations')
plt.ylabel('MSE training loss')
plt.tight_layout()
plt.savefig('FA_quad_log_tr_loss.png')
# plt.show()

np.savez('FA_quad_logs.npz', log_loss=np.array(log_loss))


# generate new points for new plots and accuracy measure 
num_tt_data = 10_000
x_tt = jnp.linspace(-1, 1, num_tt_data)
x_tt = x_tt.reshape((num_tt_data, dim_input)) # Shape: (# data, input dimension)
y_tt = f(x_tt)

y_NN = batched_prediction(params, x_tt)

relative_l2_error = jnp.sqrt(jnp.sum((y_NN - y_tt)**2)/jnp.sum(y_tt**2))
print(f'relative l2 error = {relative_l2_error:.2e}')
np.savez('FA_quad_logs.npz', log_loss=np.array(log_loss), rel_l2_err=relative_l2_error)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
axes[0].plot(x_tt,y_tt,label='target')
axes[0].plot(x_tt,y_NN,label='NN prediction')
axes[0].set_xlabel('x')
axes[0].set_ylabel('NN prediction')
axes[0].legend()
axes[1].semilogy(x_tt,jnp.abs(y_NN-y_tt))
axes[1].set_xlabel('x')
axes[1].set_ylabel('absolute error')
plt.suptitle(f'Visualizataion of the trained model: DNN ({layers[0]:d}-{layers[1]:d}-{layers[2]:d}-{layers[3]:d}), GD w/ lr={lr:.1e} over {maxITER//1000}K iterations')
plt.tight_layout()
plt.savefig('FA_quad_Graph_and_Error.png')
# plt.show()


savez_dict = dict()
for i in range(len(params)):
    savez_dict[f'w{i}'] = params[i][0]
    savez_dict[f'b{i}'] = params[i][1]
np.savez('FA_quad_model.npz', **savez_dict)