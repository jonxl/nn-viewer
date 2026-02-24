"""ODE results visualizer for real PINN training data."""

import csv
import json
import numpy as np

from math import factorial
from typing import Dict, List, Optional, Tuple

from visualizer import GeneralizedVisualizer, PlotConfig
from ui import RangeSliderConfig
from theme import Theme


class ODEResultsVisualizer(GeneralizedVisualizer):
  """
  Visualizer for real ODE PINN training results.

  Displays training progression of a single PINN run across iterations,
  with comparisons to both the benchmark power series and the exact
  analytical ODE solution.
  """

  def __init__(
    self,
    results_json_path: str,
    loss_csv_path: str,
    x_range: Tuple[float, float] = (0, 1),
    num_points: int = 1000,
    initial_iteration: int = 1000,
    theme: Optional[Theme] = None,
  ):
    # Load results.json -> iteration_to_snapshot dict
    with open(results_json_path, "r") as f:
      data = json.load(f)

    metadata = data["metadata"]
    snapshots = data["milestones"]

    self.iteration_to_snapshot: Dict[int, dict] = {
      entry["iteration"]: entry for entry in snapshots
    }

    # Extract constant data from metadata
    self.benchmark_coefficients = np.array(metadata["benchmark_coefficients"])
    self.alpha_matrix = metadata["ode_matrix"]

    # Load loss CSV
    self.loss_data = self._load_loss_csv(loss_csv_path)

    # Precompute x data
    self.x_data = np.linspace(x_range[0], x_range[1], num_points)

    # # Compute analytical ODE solution (static)
    # self._setup_analytical_solution()
    # self.analytical_solution = self._compute_analytical_solution(self.x_data)
    self.analytical_solution = None

    # Compute benchmark power series (static)
    self.benchmark_series = self._evaluate_factorial_power_series(
      self.benchmark_coefficients, self.x_data
    )

    # Determine iteration range from data
    iterations = sorted(self.iteration_to_snapshot.keys())
    iter_min = iterations[0]
    iter_max = iterations[-1]
    iter_step = iterations[1] - iterations[0] if len(iterations) > 1 else 100

    # Slider config — range slider for iteration window [start, end]
    slider_configs = [
      RangeSliderConfig(
        name="iteration",
        label="Iteration",
        valmin=iter_min,
        valmax=iter_max,
        valinit=(iter_min, initial_iteration),
        valstep=iter_step,
      ),
    ]

    # Plot configs
    plot_configs = self._create_plot_configs()

    # Build minimal data_dict (real routing happens in _get_plot_data)
    data_dict = {}

    super().__init__(
      data_dict=data_dict,
      plot_configs=plot_configs,
      slider_configs=slider_configs,
      layout=(4, 2),
      figsize=(20, 16),
      main_title="PINN ODE Training Analysis",
      hide_empty_plots=True,
      theme=theme,
    )

  @staticmethod
  def _load_loss_csv(path: str) -> Dict[str, np.ndarray]:
    """Load loss data from CSV file."""
    iterations = []
    total = []
    bc = []
    pde = []
    supervised = []
    with open(path, "r") as f:
      reader = csv.reader(f)
      next(reader)  # skip header
      for row in reader:
        if not row or not row[0].strip():
          continue
        iterations.append(int(row[0]))
        total.append(float(row[1]))
        bc.append(float(row[2]))
        pde.append(float(row[3]))
        supervised.append(float(row[4]))
    return {
      "iteration": np.array(iterations),
      "total": np.array(total),
      "bc": np.array(bc),
      "pde": np.array(pde),
      "supervised": np.array(supervised),
    }

  # def _setup_analytical_solution(self):
  #   """Set up the analytical solution from the ODE coefficients."""
  #   # ODE: u'' + a1*u' + a0*u = 0  (leading coefficient a2=1 is implicit)
  #   a1, a0 = self.alpha_matrix
  #   # Characteristic equation: r^2 + a1*r + a0 = 0
  #   discriminant = a1**2 - 4 * a0
  #
  #   sqrt_disc = np.sqrt(abs(discriminant))
  #   self.r1 = (-a1 + sqrt_disc) / 2
  #   self.r2 = (-a1 - sqrt_disc) / 2
  #
  #   # Initial conditions from benchmark coefficients
  #   u0 = float(self.benchmark_coefficients[0])
  #   u_prime0 = float(self.benchmark_coefficients[1])
  #
  #   # Solve: c1 + c2 = u0, c1*r1 + c2*r2 = u_prime0
  #   self.c1 = (u_prime0 - u0 * self.r2) / (self.r1 - self.r2)
  #   self.c2 = u0 - self.c1

  # def _compute_analytical_solution(self, x: np.ndarray) -> np.ndarray:
  #   """Compute exact analytical ODE solution at points x."""
  #   return self.c1 * np.exp(self.r1 * x) + self.c2 * np.exp(self.r2 * x)

  def _evaluate_factorial_power_series(
    self, coefficients: np.ndarray, x: np.ndarray
  ) -> np.ndarray:
    """Evaluate factorial-normalized power series.

    u(x) = sum(coeff[i] * x^i / i!) for i in 0..N-1

    Matches Julia: sum(a[i] * x^(i-1) / fact[i] for i in 1:N)
    """
    result = np.zeros_like(x, dtype=float)
    for i, coeff in enumerate(coefficients):
      result += coeff * (x**i) / factorial(i)
    return result

  def _create_plot_configs(self) -> List[PlotConfig]:
    """Create the 8 plot configurations."""
    return [
      # Row 1: Function Analysis
      PlotConfig(
        data_key="function_comparison",
        title="ODE Solution Comparison",
        xlabel="x",
        ylabel="u(x)",
        plot_type="line",
        colors=["#66bb6a", "#ff8a65", "#4fc3f7"],
        linestyles=["-.", "-", "--"],
        labels=["Benchmark Series", "PINN Series", "Analytical Solution"],
      ),
      PlotConfig(
        data_key="function_error",
        title="Absolute Error of Solution",
        xlabel="x",
        ylabel="|Error|",
        plot_type="semilogy",
        colors=["#81c784"],
        labels=["PINN Error"],
      ),
      # Row 2: Coefficient Analysis
      PlotConfig(
        data_key="coefficient_comparison",
        title="Coefficient Comparison",
        xlabel="Coefficient Index",
        ylabel="Coefficient Value",
        plot_type="line",
        colors=["#4fc3f7", "#ff8a65"],
        linestyles=["-", "-"],
        labels=["Benchmark", "PINN"],
      ),
      PlotConfig(
        data_key="coefficient_error",
        title="Coefficient Error",
        xlabel="Coefficient Index",
        ylabel="|Error|",
        plot_type="semilogy",
        colors=["#ef5350"],
        labels=["|Benchmark - PINN|"],
      ),
      # Row 3-4: Loss Plots
      PlotConfig(
        data_key="loss_total",
        title="Total Loss",
        xlabel="Iteration",
        ylabel="Loss",
        plot_type="semilogy",
        colors=["#e0e0e0"],
        labels=["Total Loss"],
      ),
      PlotConfig(
        data_key="loss_bc",
        title="BC Loss",
        xlabel="Iteration",
        ylabel="Loss",
        plot_type="semilogy",
        colors=["#ef5350"],
        labels=["BC Loss"],
      ),
      PlotConfig(
        data_key="loss_pde",
        title="PDE Loss",
        xlabel="Iteration",
        ylabel="Loss",
        plot_type="semilogy",
        colors=["#42a5f5"],
        labels=["PDE Loss"],
      ),
      PlotConfig(
        data_key="loss_supervised",
        title="Supervised Loss",
        xlabel="Iteration",
        ylabel="Loss",
        plot_type="semilogy",
        colors=["#66bb6a"],
        labels=["Supervised Loss"],
      ),
    ]

  def _get_plot_data(self, data_key: str):
    """Retrieve data for plotting based on current iteration range slider."""
    iter_start, iter_end = self.slider_values["iteration"]
    iter_start = int(round(iter_start))
    iter_end = int(round(iter_end))

    # Non-loss plots use the right handle for snapshot selection
    snapshot = self.iteration_to_snapshot.get(iter_end)

    if data_key == "function_comparison":
      if snapshot is None:
        return None
      pinn_coeffs = np.array(snapshot["coefficients"])
      pinn_series = self._evaluate_factorial_power_series(pinn_coeffs, self.x_data)
      result = {
        "benchmark": {"x": self.x_data, "y": self.benchmark_series},
        "pinn": {"x": self.x_data, "y": pinn_series},
      }
      if self.analytical_solution is not None:
        result["analytical"] = {"x": self.x_data, "y": self.analytical_solution}
      return result

    elif data_key == "function_error":
      if snapshot is None:
        return None
      pinn_coeffs = np.array(snapshot["coefficients"])
      pinn_series = self._evaluate_factorial_power_series(pinn_coeffs, self.x_data)
      reference = self.analytical_solution if self.analytical_solution is not None else self.benchmark_series
      error = np.abs(reference - pinn_series)
      return {"error": {"x": self.x_data, "y": error}}

    elif data_key == "coefficient_comparison":
      if snapshot is None:
        return None
      pinn_coeffs = np.array(snapshot["coefficients"])
      min_len = min(len(self.benchmark_coefficients), len(pinn_coeffs))
      indices = np.arange(min_len)
      return {
        "benchmark": {"x": indices, "y": self.benchmark_coefficients[:min_len]},
        "pinn": {"x": indices, "y": pinn_coeffs[:min_len]},
      }

    elif data_key == "coefficient_error":
      if snapshot is None:
        return None
      pinn_coeffs = np.array(snapshot["coefficients"])
      min_len = min(len(self.benchmark_coefficients), len(pinn_coeffs))
      indices = np.arange(min_len)
      error = np.abs(self.benchmark_coefficients[:min_len] - pinn_coeffs[:min_len])
      return {"error": {"x": indices, "y": error}}

    elif data_key.startswith("loss_"):
      loss_type = data_key[5:]  # "total", "bc", "pde", "supervised"
      if loss_type not in self.loss_data:
        return None
      # Window: show loss within [iter_start, iter_end]
      mask = (
        (self.loss_data["iteration"] >= iter_start)
        & (self.loss_data["iteration"] <= iter_end)
      )
      x_vals = self.loss_data["iteration"][mask]
      y_vals = self.loss_data[loss_type][mask]
      return {loss_type: {"x": x_vals, "y": y_vals}}

    return None

  def _update_plot(self, ax, config, plot_idx):
    """Override to pin x-axis limits on loss plots to the iteration window."""
    super()._update_plot(ax, config, plot_idx)
    if config.data_key.startswith("loss_") and ax.get_visible():
      iter_start, iter_end = self.slider_values["iteration"]
      ax.set_xlim(int(round(iter_start)), int(round(iter_end)))
