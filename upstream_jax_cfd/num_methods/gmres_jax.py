import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnames=["max_iter", "M_func","A_func"])
def gmres(A_func, b, x0=None, tol=1e-8, max_iter=50, M_func=None):
    """
    Matrix-free GMRES solver using JAX.

    Parameters:
        A_func: function that computes A @ x
        b: right-hand side vector
        x0: initial guess (default is zero)
        tol: relative residual tolerance
        max_iter: maximum number of iterations (static for JIT)
        M_func: preconditioner function (default is identity)

    Returns:
        x: approximate solution
        info: 0 if converged, else number of iterations
    """
    n = b.shape[0]
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if M_func is None:
        M_func = lambda x: x

    # Initial residual: r0 = M^{-1}(b - A x0)
    r0 = M_func(b - A_func(x0))
    beta = jnp.linalg.norm(r0)

    # e1 = [beta, 0, 0, ..., 0] (unit vector scaled by beta)
    e1 = jnp.zeros((max_iter + 1,))
    e1 = e1.at[0].set(beta)

    # Arnoldi basis vectors: V[:, i]
    V = jnp.zeros((n, max_iter + 1))
    V = V.at[:, 0].set(r0 / beta)

    # Hessenberg matrix H (upper Hessenberg)
    H = jnp.zeros((max_iter + 1, max_iter))

    def arnoldi_step(V, H, k):
        """Performs 1 step of the Arnoldi process."""
        v_k = V[:, k]
        w = M_func(A_func(v_k))

        # Modified Gram-Schmidt
        h_col = jnp.zeros((max_iter + 1,))  # full-length column to avoid shape issues

        def orthogonalize(j, val):
            w, h_col = val
            vj = V[:, j]
            h = jnp.dot(vj, w)
            w = w - h * vj
            h_col = h_col.at[j].set(h)
            return w, h_col

        w, h_col = lax.fori_loop(0, k + 1, orthogonalize, (w, h_col))

        h_next = jnp.linalg.norm(w)
        v_next = jnp.where(h_next > 0, w / h_next, jnp.zeros_like(w))

        V = V.at[:, k + 1].set(v_next)
        H = H.at[:, k].set(h_col)
        H = H.at[k + 1, k].set(h_next)

        return V, H

    def body_fn(state, k):
        """One GMRES iteration step."""
        V, H, x0 = state

        # Get v_k
        vk = V[:, k]
        z = M_func(A_func(vk))

        # Perform Arnoldi step to expand Krylov subspace
        V, H = arnoldi_step(V, H, k, z)

        # Solve least-squares problem: minimize ||beta*e1 - H_k y||
        # Use static slicing first, then dynamic indexing
        Hk_full = H[:max_iter + 1, :max_iter]  # static shape
        Hk = Hk_full[:k + 2, :k + 1]

        e1k = e1[:k + 2]  # slicing into static array is OK

        y, *_ = jnp.linalg.lstsq(Hk, e1k, rcond=None)

        # Update solution approximation: x_k = x0 + V_k @ y
        Vk = V[:, :k + 1]  # static + dynamic slicing is safe
        xk = x0 + Vk @ y

        # Compute true residual norm ||b - A x_k||
        r_norm = jnp.linalg.norm(b - A_func(xk))

        return (V, H, xk), r_norm

    def scan_body(carry, k):
        state, _ = carry
        state, res_norm = body_fn(state, k)
        return (state, res_norm), res_norm

    # Run GMRES iterations using lax.scan
    init_state = (V, H, x0)
    (_, _, x_final), res_norms = lax.scan(scan_body, (init_state, beta), jnp.arange(max_iter))

    final_res_norm = res_norms[-1]
    info = jnp.where(final_res_norm <= tol, 0, max_iter)

    return x_final, info
