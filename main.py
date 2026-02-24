import argparse

from visualizer import setup_backend
from views import ODEResultsVisualizer


def main():
    parser = argparse.ArgumentParser(description="PINN ODE Training Visualizer")
    parser.add_argument("--results", required=True, help="Path to results JSON file")
    parser.add_argument("--loss", required=True, help="Path to loss CSV file")
    args = parser.parse_args()

    setup_backend()

    visualizer = ODEResultsVisualizer(
        results_json_path=args.results,
        loss_csv_path=args.loss,
        x_range=(0, 1),
        num_points=1000,
        initial_iteration=1000,
    )
    visualizer.show()


if __name__ == "__main__":
    main()
