import tensorflow_probability.substrates.jax as tfp
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from functools import partial
import numpy as np

dist = tfp.distributions

class seq2point(nn.Module):
    @nn.compact
    def __call__(self, X, deterministic):
        X = nn.Conv(30, kernel_size=(10,))(X)
        X = nn.relu(X)
        X = nn.Conv(30, kernel_size=(8,))(X)
        X = nn.relu(X)        
        X = nn.Conv(40, kernel_size=(6,))(X)
        X = nn.relu(X)
        X = nn.Conv(50, kernel_size=(5,))(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=0.2, deterministic=deterministic)(X)
        X = nn.Conv(50, kernel_size=(5,))(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=0.2, deterministic=deterministic)(X)
        X = X.reshape((X.shape[0], -1))
        X = nn.Dense(1024)(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=0.2, deterministic=deterministic)(X)
        sigma = nn.softplus(nn.Dense(5)(X))
        # add task-specific layers
        mean1 = nn.Dense(64)(X)
        mean1 = nn.relu(mean1)
        mean1 = nn.Dense(1)(mean1)
        # sigma1 = nn.softplus(nn.Dense(1)(mean1))

        mean2 = nn.Dense(64)(X)
        mean2 = nn.relu(mean2)
        mean2 = nn.Dense(1)(mean2)
        # sigma2 = nn.softplus(nn.Dense(1)(mean2))

        mean3 = nn.Dense(64)(X)
        mean3 = nn.relu(mean3)
        mean3 = nn.Dense(1)(mean3)
        # sigma3 = nn.softplus(nn.Dense(1)(mean3))

        mean4 = nn.Dense(64)(X)
        mean4 = nn.relu(mean4)
        mean4 = nn.Dense(1)(mean4)
        # sigma4 = nn.softplus(nn.Dense(1)(mean4))

        mean5 = nn.Dense(64)(X)
        mean5 = nn.relu(mean5)
        mean5 = nn.Dense(1)(mean5)
        # sigma5 = nn.softplus(nn.Dense(1)(mean5))

        mean = jnp.concatenate([mean1, mean2, mean3, mean4, mean5], axis = 1)
        # sigma = jnp.concatenate([sigma1, sigma2, sigma3, sigma4, sigma5], axis = 1)
        return mean, sigma

    def loss_fn(self, params, X, y, deterministic=False, rng=jax.random.PRNGKey(0)):
        mean, sigma = self.apply(
            params, X, deterministic=deterministic, rngs={"dropout": rng}
        )

        def loss(mean, sigma, y):
            d = dist.Normal(loc=mean, scale=sigma)
            return -np.mean(jnp.sum(d.log_prob(y), axis=-1))

        return jnp.mean(jax.vmap(loss, in_axes=(0, 0, 0))(mean, sigma, y))
    
def fit(
    model,
    params,
    X,
    y,
    deterministic,
    batch_size=32,
    learning_rate=0.01,
    epochs=10,
    rng=jax.random.PRNGKey(0),
):
    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(params)

    loss_fn = partial(model.loss_fn, deterministic=deterministic)
    loss_grad_fn = jax.value_and_grad(loss_fn)
    losses = []
    total_epochs = (len(X) // batch_size) * epochs

    carry = {}
    carry["params"] = params
    carry["state"] = opt_state

    @jax.jit
    def one_epoch(carry, rng):
        params = carry["params"]
        opt_state = carry["state"]
        idx = jax.random.choice(
            rng, jnp.arange(len(X)), shape=(batch_size,), replace=False
        )
        loss_val, grads = loss_grad_fn(params, X[idx], y[idx], rng=rng)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        carry["params"] = params
        carry["state"] = opt_state

        return carry, loss_val

    carry, losses = jax.lax.scan(one_epoch, carry, jax.random.split(rng, total_epochs))
    return carry["params"], losses