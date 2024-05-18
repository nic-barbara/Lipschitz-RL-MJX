import jax
import jax.numpy as jnp
import optax

from jax import lax
from typing import Any, Callable
from liprl.networks.typing import PrecisionLike
# from flax.typing import PrecisionLike


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def l2_norm(x, eps=jnp.finfo(jnp.float32).eps, **kwargs):
    """Compute l2 norm of a vector/matrix with JAX.
    This is safe for backpropagation, unlike `jnp.linalg.norm`."""
    return jnp.sqrt(jnp.maximum(jnp.sum(x**2, **kwargs), eps))


def cayley(W, return_split=False):
    """Perform Cayley transform on a stacked matrix [U; V]"""
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    U, V = W[:n, :], W[n:, :]
    Z = (U - U.T) + (V.T @ V)
    I = jnp.eye(n)
    ZI = Z + I
    
    # Note that B * A^-1 = solve(A.T, B.T).T
    A_T = jnp.linalg.solve(ZI, I-Z)
    B_T = -2 * jnp.linalg.solve(ZI.T, V.T).T
    
    if return_split:
        return A_T, B_T
    return jnp.concatenate([A_T, B_T])


def dot_lax(input1, input2, precision: PrecisionLike = None):
    """Wrapper around lax.dot_general(). Use this instead of `@` for
    more efficient array-matrix multiplication and backpropagation.
    
    NOTE: `@` might actually just default back to this anyway. Look
    into this later.
    """
    return lax.dot_general(
        input1,
        input2,
        (((input1.ndim - 1,), (1,)), ((), ())),
        precision=precision,
    )


def estimate_lipschitz_lower(    
    policy,
    n_inputs,
    batches=128,
    max_iter=450,
    learning_rate=0.01,
    clip_at=0.01,
    init_var=0.001,
    verbose=True,
    seed=0
):
    """
    Estimate a lower-bound on the Lipschitz constant with gradient descent.
    
    NOTE: this function is written for static models. See Julia version
          for code that will work on dynamic models (eg: LSTM).
          
    https://github.com/nic-barbara/CDC2023-YoulaREN/blob/main/src/Robustness/utils.jl
    """
    
    # Initialise model inputs
    key = jax.random.PRNGKey(seed)
    key, rng1, rng2 = jax.random.split(key, 3)
    u1 = init_var * jax.random.normal(rng1, (batches, n_inputs))
    u2 = u1 + 1e-4 * jax.random.normal(rng2, (batches, n_inputs))

    # Set up optimization parameters
    params = (u1, u2)

    # Optimizer
    scheduler = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=150,
        decay_rate=0.1,
        end_value=0.001*learning_rate,
        staircase=True
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_at),
        optax.inject_hyperparams(optax.adam)(learning_rate=scheduler),
        optax.scale(-1.0) # To maximise the Lipschitz bound
    )
    
    optimizer_state = optimizer.init(params)

    # Loss function
    def lip_loss(params):
        u1, u2 = params
        key = jax.random.PRNGKey(0)
        key, rng1, rng2 = jax.random.split(key, 3)
        
        y1, _ = policy(u1, rng1)
        y2, _ = policy(u2, rng2)
        
        return l2_norm(y2 - y1) / l2_norm(u1 - u2)

    # Gradient of the loss function
    grad_loss = jax.grad(lip_loss)
    jit_lip_loss = jax.jit(lip_loss)
    jit_grad_loss = jax.jit(grad_loss)

    # Use gradient descent to estimate the Lipschitz bound
    lips = []
    for iter in range(max_iter):
        
        grad_value = jit_grad_loss(params)
        updates, optimizer_state = optimizer.update(grad_value, optimizer_state)
        params = optax.apply_updates(params, updates)
        
        lips.append(jit_lip_loss(params))
        if verbose and iter % 10 == 0:
            print("Iter: ", iter, "\t L: ", lips[-1], "\t η: ", 
                  optimizer_state[1].hyperparams['learning_rate'])
    
    return max(lips)
    
