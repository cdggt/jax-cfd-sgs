import jax.numpy as jnp
import jax_cfd.num_methods.gmres_jax as jnm
import jax as jax

A = jnp.array([[4.0, 1.0], [1.0, 3.0]])

def A_func(x):
    return A @ x

def M_func(x):
    M_inv = 1.0 / jnp.diag(A)
    return M_inv * x

b = jnp.array([1.0, 2.0])
x0 = jnp.zeros_like(b)

x, info = jnm.gmres(A_func, b, x0, tol=1e-10, M_func=M_func)

print("Solution:", x)
print("Info:", info)
print("Residual norm:", jnp.linalg.norm(b - A_func(x)))