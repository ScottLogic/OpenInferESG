"""
RAGAS Visualization Module
-------------------------
Functions for generating visualizations of RAGAS evaluation results.
"""

import json
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


def generate_bar_chart(json_file_path: str) -> Optional[str]:
    """
    Generate a bar chart based on the evaluation results in a JSON file

    Args:
        json_file_path: Path to the JSON file with evaluation results

    Returns:
        Path to the saved chart image or None if chart generation fails
    """

    try:
        # Load JSON data and convert to DataFrame
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # Handle the "AVERAGE" row - exclude it from plotting
        avg_row = df[df["question"] == "AVERAGE"]
        plot_df = df[df["question"] != "AVERAGE"]

        # Get metrics columns (all columns except 'question')
        metrics = [col for col in plot_df.columns if col != "question"]

        # Filter out metrics that have all None values
        valid_metrics = []
        for metric in metrics:
            # Use pandas methods safely with explicit checking
            metric_data = plot_df[metric]
            if isinstance(metric_data, pd.Series):
                if metric_data.notna().any():
                    valid_metrics.append(metric)

        metrics = valid_metrics

        if not metrics:
            print("No valid metrics found with non-null values for plotting")
            return None

        # Keep all questions regardless of count
        question_count = len(plot_df)
        print(f"Plotting all {question_count} questions")

        # Set up the plot with larger figure for many questions
        question_count = len(plot_df)
        # Dynamically adjust figure width based on number of questions
        fig_width = max(12, question_count * 0.3)
        plt.figure(figsize=(fig_width, 10))
        width = 0.8 / len(metrics)  # Width for each metric bar

        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            positions = [j + width * i for j in range(len(plot_df))]
            # Handle None/null values by replacing them with NaN
            metric_data = plot_df[metric]
            if isinstance(metric_data, pd.Series):
                values = metric_data.map(lambda x: float("nan") if x is None else x).tolist()
            else:
                # Convert to list and handle None values
                values = [float("nan") if val is None else val for val in metric_data]
            # Filter out NaN values for plotting
            valid_positions = [positions[j] for j in range(len(values)) if not pd.isna(values[j])]
            valid_values = [values[j] for j in range(len(values)) if not pd.isna(values[j])]
            plt.bar(valid_positions, valid_values, width=width, label=metric)

        # Set up x-axis with question labels
        question_count = len(plot_df)
        question_labels = [f"Q{i + 1}" for i in range(question_count)]
        tick_positions = [j + width * (len(metrics) - 1) / 2 for j in range(question_count)]

        if tick_positions:  # Make sure we have tick positions before setting them
            # If we have many questions, adjust the rotation and font size
            if question_count > 20:
                plt.xticks(tick_positions, question_labels, rotation=90, fontsize=8, ha="center")
            else:
                plt.xticks(tick_positions, question_labels, rotation=45, ha="right")

        # Add average lines if available
        if len(avg_row) > 0:  # Check length instead of using empty property
            for i, metric in enumerate(metrics):
                if metric in avg_row.columns:
                    try:
                        # Get first value safely
                        metric_series = avg_row[metric]
                        if len(metric_series) > 0:
                            # Access first element using integer indexing
                            avg_value = metric_series.iat[0] if isinstance(metric_series, pd.Series) else None
                            # Check if it's not a NaN value or None
                            if avg_value is not None and not pd.isna(avg_value) and isinstance(avg_value, (int, float)):
                                plt.axhline(y=avg_value, color=f"C{i}", linestyle="--", alpha=0.7)
                                # Add a text label with the average value
                                plt.text(
                                    len(plot_df) - 0.5,
                                    avg_value + 0.02,
                                    f"Avg: {avg_value:.2f}",
                                    color=f"C{i}",
                                    fontsize=9,
                                    ha="right",
                                )
                    except Exception as e:
                        print(f"Error adding average line for {metric}: {e}")

        # Format and save the plot
        plt.ylabel("Score (0-1)")
        plt.title(f"RAGAS Evaluation Results\n{', '.join(metrics)}")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        # For many questions, make more room at the bottom for labels
        question_count = len(plot_df)
        if question_count > 20:
            plt.subplots_adjust(bottom=0.2)

        # Add a horizontal line at 0.5 for reference
        plt.axhline(y=0.5, color="gray", linestyle="-", alpha=0.3)

        plt.tight_layout()

        # Position legend based on number of metrics
        legend_cols = min(len(metrics), 3)  # Max 3 columns for readability
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=legend_cols, frameon=True, fancybox=True, shadow=True
        )

        # Save and close
        output_image_path = json_file_path.replace(".json", "_chart.png")
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_image_path

    except Exception as e:
        print(f"Error generating chart: {e}")
        return None
