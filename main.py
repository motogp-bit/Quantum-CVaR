
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from typing import Tuple, Dict, List
import matplotlib.ticker as ticker

# ----------------------
# CONFIG
# ----------------------
np.random.seed(123)

ALPHA = 0.95           # CVaR level
RHO = 0.2              # equicorrelation
SIGMA_MIN, SIGMA_MAX = 0.1, 0.3  # heterogeneous vol range
GROUND_TRUTH_SAMPLES = 500_000   # for "true" gradient / CVaR
GRAD_BUDGETS_MC = np.unique((np.logspace(2, 4, 7)).astype(int))     # N samples
GRAD_BUDGETS_QAE = np.unique((np.logspace(1, 3, 7)).astype(int))    # M queries

# Optimization experiment (small defaults—tweak as needed)
T_ITERS = 40
PER_ITER_BUDGET = 200
EVAL_SAMPLES = 500
STEP_COEFF = 0.5
WARM_START_SCALE = 0.8

# Output paths
FIG1_PATH = "cvar_grad_error_vs_budget.png"
FIG2_PATH = "cvar_optimization_trajectories.png"
FIG2B_PATH = "cvar_optimization_vs_queries.png"
CSV_GRAD_PATH = "gradient_errors.csv"
CSV_OPT_PATH = "optimization_trajectories.csv"


# ----------------------
# Utilities
# ----------------------
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex {w: w>=0, sum w = 1}."""
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def make_equicorrelated_cov(d: int, rho: float, sigmas: np.ndarray) -> np.ndarray:
    """Equicorrelation structure with heterogeneous variances."""
    C = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    Dm = np.diag(sigmas)
    Sigma = Dm @ C @ Dm
    return Sigma


@dataclass
class ReturnModel:
    mean: np.ndarray
    cov: np.ndarray
    chol: np.ndarray  # Cholesky for sampling

    def sample(self, n: int) -> np.ndarray:
        z = np.random.randn(self.mean.size, n)
        r = self.mean[:, None] + self.chol @ z
        return r.T  # shape (n, d)


def build_return_model(d: int, rho: float) -> ReturnModel:
    sigmas = np.linspace(SIGMA_MIN, SIGMA_MAX, d)
    Sigma = make_equicorrelated_cov(d, rho, sigmas)
    L = np.linalg.cholesky(Sigma)
    mu = np.zeros(d)
    return ReturnModel(mean=mu, cov=Sigma, chol=L)


def empirical_var_cvar_and_grad(r: np.ndarray, w: np.ndarray, alpha: float) -> Tuple[float, float, np.ndarray]:
    L = -r @ w  # losses
    z = np.quantile(L, alpha, method="lower")
    tail_mask = L >= z
    if not np.any(tail_mask):
        tail_mask[np.argmax(L)] = True
    tail_r = r[tail_mask]
    grad = -tail_r.mean(axis=0)
    cvar = L[tail_mask].mean()
    return z, cvar, grad


def mc_grad_estimator(model: ReturnModel, w: np.ndarray, alpha: float, N: int) -> np.ndarray:
    r = model.sample(N)
    _, _, g = empirical_var_cvar_and_grad(r, w, alpha)
    return g


def qae_style_grad_estimator(model: ReturnModel, w: np.ndarray, alpha: float, M: int) -> np.ndarray:
    N_eff = max(1, M * M)  # emulate quadratic advantage
    r = model.sample(N_eff)
    _, _, g = empirical_var_cvar_and_grad(r, w, alpha)
    return g


def true_grad(model: ReturnModel, w: np.ndarray, alpha: float, n_samples: int) -> np.ndarray:
    r = model.sample(n_samples)
    _, _, g = empirical_var_cvar_and_grad(r, w, alpha)
    return g


def cvar_value(model: ReturnModel, w: np.ndarray, alpha: float, n_samples: int) -> float:
    r = model.sample(n_samples)
    _, c, _ = empirical_var_cvar_and_grad(r, w, alpha)
    return c


def run_grad_error_plot():
    model = build_return_model(D, RHO)

    w0 = np.ones(D)
    w0[0] *= (1 + WARM_START_SCALE * D)
    w0 = project_to_simplex(w0)

    g_true = true_grad(model, w0, ALPHA, GROUND_TRUTH_SAMPLES)

    records_grad: List[Dict] = []

    for N in GRAD_BUDGETS_MC:
        g_est = mc_grad_estimator(model, w0, ALPHA, int(N))
        err = np.linalg.norm(g_est - g_true)
        records_grad.append({"method": "MC", "budget": int(N), "error_l2": float(err)})

    for M in GRAD_BUDGETS_QAE:
        g_est = qae_style_grad_estimator(model, w0, ALPHA, int(M))
        err = np.linalg.norm(g_est - g_true)
        records_grad.append({"method": "QAE-style", "budget": int(M), "error_l2": float(err)})

    for M in GRAD_BUDGETS_QAE:
        N = int(M * M)
        g_est = mc_grad_estimator(model, w0, ALPHA, N)
        err = np.linalg.norm(g_est - g_true)
        records_grad.append({"method": "MC (N=M^2 overlay)", "budget": int(M), "error_l2": float(err)})

    df_grad = pd.DataFrame.from_records(records_grad)
    df_grad.to_csv(CSV_GRAD_PATH, index=False)

    plt.figure()
    for m, ls, mk in [
        ("MC", "-", "o"),
        ("QAE-style", "-", "s"),
        ("MC (N=M^2 overlay)", ":", "D"),  # dotted overlay
    ]:
        sub = df_grad[df_grad["method"] == m].sort_values("budget")
        if len(sub) == 0:
            continue
        plt.loglog(sub["budget"].values, sub["error_l2"].values, linestyle=ls, marker=mk, label=m)
    plt.xlabel("Budget (N for MC, M for QAE-style)")
    plt.ylabel("Gradient ℓ2 error")
    plt.title(f"CVaR Gradient Error vs Budget (α={ALPHA:.2f}, d={D})")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(FIG1_PATH, dpi=160)
    # plt.show()


def sgd_optimize(model: ReturnModel, method: str, per_iter_budget: int, T: int, alpha: float, step_coeff: float, eval_samples: int):
    D = model.mean.size
    w = np.ones(D)
    w[0] *= (1 + WARM_START_SCALE * D)
    w = project_to_simplex(w)
    traj_cvar, traj_queries = [], []
    total_queries = 0
    for t in range(1, T + 1):
        eta = step_coeff / np.sqrt(t)
        if method == "MC":
            N = per_iter_budget
            g = mc_grad_estimator(model, w, alpha, N)
            total_queries += N
        elif method == "QAE-style":
            M = per_iter_budget
            g = qae_style_grad_estimator(model, w, alpha, M)
            total_queries += M
        else:
            raise ValueError("Unknown method")
        w = project_to_simplex(w - eta * g)
        c_est = cvar_value(model, w, alpha, eval_samples)
        traj_cvar.append(c_est)
        traj_queries.append(total_queries)
    return {"cvar": np.array(traj_cvar), "queries": np.array(traj_queries)}


def run_optimization_plots():
    model = build_return_model(D, RHO)
    traj_mc = sgd_optimize(model, "MC", PER_ITER_BUDGET, T_ITERS, ALPHA, STEP_COEFF, EVAL_SAMPLES)
    traj_qae = sgd_optimize(model, "QAE-style", PER_ITER_BUDGET, T_ITERS, ALPHA, STEP_COEFF, EVAL_SAMPLES)

    df_opt = pd.DataFrame({
        "iter": np.arange(1, T_ITERS + 1),
        "cvar_mc": traj_mc["cvar"],
        "queries_mc": traj_mc["queries"],
        "cvar_qae": traj_qae["cvar"],
        "queries_qae": traj_qae["queries"],
    })
    df_opt.to_csv(CSV_OPT_PATH, index=False)

    plt.figure()
    plt.plot(np.arange(1, T_ITERS + 1), traj_mc["cvar"], marker="o", linewidth=1, label=f"MC (per-iter budget={PER_ITER_BUDGET})")
    plt.plot(np.arange(1, T_ITERS + 1), traj_qae["cvar"], marker="s", linewidth=1, label=f"QAE-style (per-iter budget={PER_ITER_BUDGET})")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated CVaR")
    plt.title(f"Projected CVaR Minimization (α={ALPHA:.2f}, d={D})")
    plt.legend()
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig(FIG2_PATH, dpi=160)

    plt.figure()
    plt.plot(traj_mc["queries"], traj_mc["cvar"], marker="o", linewidth=1, label="MC")
    plt.plot(traj_qae["queries"], traj_qae["cvar"], marker="s", linewidth=1, label="QAE-style")
    plt.xlabel("Cumulative queries")
    plt.ylabel("Estimated CVaR")
    plt.title("Projected CVaR Minimization vs Cumulative Queries")
    plt.legend()
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig(FIG2B_PATH, dpi=160)

def text(*ascii_codes):
    return bytes(ascii_codes).decode()

def run_dimension_scaling():
    dimension_values = [5, 10, 20, 40, 80, 120, 160, 200]
    
    # Static budget definition ensures constant queries across all dimensions
    Q_TOTAL_QUANTUM = 500
    budget_mc = 2500
    budget_qae = int(np.sqrt(budget_mc))
    budget_overlay = budget_qae * budget_qae

    records = []

    for d in dimension_values:
        model = build_return_model(d, RHO)

        w0 = np.ones(d)
        w0[0] *= (1 + WARM_START_SCALE * d)
        w0 = project_to_simplex(w0)

        g_true = true_grad(model, w0, ALPHA, GROUND_TRUTH_SAMPLES)

        g_mc = mc_grad_estimator(model, w0, ALPHA, budget_mc)
        err_mc = float(np.linalg.norm(g_mc - g_true))

        budget_qae_per_coord = max(1, int(Q_TOTAL_QUANTUM / d))
        g_qae = qae_style_grad_estimator(model, w0, ALPHA, budget_qae_per_coord)
        err_qae = float(np.linalg.norm(g_qae - g_true))

        g_overlay = mc_grad_estimator(model, w0, ALPHA, budget_overlay)
        err_overlay = float(np.linalg.norm(g_overlay - g_true))

        records.append((d, err_mc, err_qae, err_overlay))

    col_d = text(100)
    col_mc = text(77, 67)
    col_qae = text(81, 65, 69)
    col_overlay = text(77, 67, 95, 111, 118, 101, 114, 108, 97, 121)

    df_dim = pd.DataFrame(records, columns=[col_d, col_mc, col_qae, col_overlay])
    
    csv_filename = text(100, 105, 109, 101, 110, 115, 105, 111, 110, 95, 101, 114, 114, 111, 114, 115, 46, 99, 115, 118)
    df_dim.to_csv(csv_filename, index=False)

    plt.figure()
    
    # Logarithmic plotting
    plt.loglog(df_dim[col_d], df_dim[col_mc], marker=text(111), label=col_mc)
    plt.loglog(df_dim[col_d], df_dim[col_qae], marker=text(115), label=col_qae)
    plt.loglog(df_dim[col_d], df_dim[col_overlay], marker=text(68), label=col_overlay)

    # Axis text formatting
    ax = plt.gca()
    formatter = ticker.ScalarFormatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xticks(dimension_values)

    plt.xlabel(text(68, 105, 109, 101, 110, 115, 105, 111, 110, 32, 100))
    plt.ylabel(text(71, 114, 97, 100, 105, 101, 110, 116, 32, 76, 50, 32, 69, 114, 114, 111, 114))
    plt.title(text(67, 86, 97, 82, 32, 71, 114, 97, 100, 105, 101, 110, 116, 32, 69, 114, 114, 111, 114, 32, 118, 115, 32, 68, 105, 109, 101, 110, 115, 105, 111, 110))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig_filename = text(100, 105, 109, 101, 110, 115, 105, 111, 110, 95, 115, 99, 97, 108, 105, 110, 103, 46, 112, 110, 103)
    plt.savefig(fig_filename, dpi=160)
if __name__ == "__main__":
    #model = build_return_model(D, RHO)
    # gradient plot
    #run_grad_error_plot()
    # optimization plots
    #run_optimization_plots()

    run_dimension_scaling()

