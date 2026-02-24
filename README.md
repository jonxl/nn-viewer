# NN Viewer

> **Disclaimer**: This project is still in testing and under active development. APIs and features may change without notice.

An interactive visualization tool for analyzing Physics-Informed Neural Networks (PINNs) power series approximations.

## Features

- **Dark/Light/High-Contrast themes** - Choose the visual style that works best for you
- **Interactive iteration slider** - Scrub through PINN training snapshots
- **ODE results visualizer** - Visualize real PINN training runs with an iteration slider, function comparison (benchmark series, PINN series), and dynamic loss history that reveals progressively as you slide through iterations
- **Multiple synchronized plots**:
  - ODE solution comparison (benchmark vs PINN)
  - Coefficient comparison
  - Coefficient and solution error plots
  - Training loss curves (total, BC, PDE, supervised)
- **Collapsible legend panel** - Toggle data series visibility with a hamburger menu (☰) that expands into toggle buttons
- **Auto-hide empty plots** - Plots with no data are automatically hidden
- **Reset button** - Restore sliders to their default values

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd nn-viewer

# Install dependencies with uv
uv sync
```

## Usage

### Quick Start

```bash
uv run python main.py --results path/to/results.json --loss path/to/loss.csv
```

### Using Themes

```python
from visualizer import GeneralizedVisualizer
from theme import get_theme

# Use a specific theme
theme = get_theme("dark")  # or "light", "high_contrast"

# Create visualizer with custom theme
visualizer = GeneralizedVisualizer(
    data_dict=data,
    plot_configs=plot_configs,
    slider_configs=slider_configs,
    theme=theme
)
```

## Project Structure

```
nn-viewer/
├── main.py              # Entry point
├── visualizer.py        # Base visualization framework (GeneralizedVisualizer, PlotConfig)
├── views/
│   ├── __init__.py      # View exports
│   └── ode_results.py   # ODEResultsVisualizer for real PINN training runs
├── theme/
│   ├── __init__.py      # Theme exports
│   └── colors.py        # Theme management (Theme class, built-in themes)
├── ui/
│   ├── __init__.py      # UI component exports
│   ├── checkbox_panel.py # Toggle button legend panel for series visibility
│   ├── slider_panel.py  # Slider panel for parameter control
│   └── button_panel.py  # Button panel for actions
├── tests/
│   ├── __init__.py      # Pytest configuration
│   ├── test_checkbox_panel.py
│   ├── test_slider_panel.py
│   └── test_theme.py
└── docs/
    └── CONTRIBUTING.md   # Contribution guidelines
```

## Controls

- **Sliders**: Use the Iteration slider to scrub through training snapshots.
- **Legend (☰)**: Collapsible toggle button panel — click the hamburger menu to expand/collapse, click individual buttons to show/hide data series
- **Reset**: Restore sliders to their initial values

## Requirements

- Python 3.10+
- matplotlib
- numpy
- PyQt5 (or another matplotlib backend: TkAgg, GTK3Agg, WXAgg)
