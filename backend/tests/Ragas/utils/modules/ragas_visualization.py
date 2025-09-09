"""
RAGAS Visualization Module
-------------------------
Functions for generating visualizations of RAGAS evaluation results.
"""
import json
from typing import Optional
import pandas as pd

# Conditionally import matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Warning: matplotlib not installed. Chart generation will not be available.")

def generate_bar_chart(json_file_path: str) -> Optional[str]:
    """
    Generate a bar chart based on the evaluation results in a JSON file

    Args:
        json_file_path: Path to the JSON file with evaluation results

    Returns:
        Path to the saved chart image or None if chart generation fails
    """
    # Check if matplotlib is available
    if plt is None:
        print("Warning: matplotlib not installed. Cannot generate charts.")
        return None

    try:
        # Load JSON data and convert to DataFrame
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # Handle the "AVERAGE" row - exclude it from plotting
        avg_row = df[df['question'] == 'AVERAGE']
        plot_df = df[df['question'] != 'AVERAGE']

        # Get metrics columns (all columns except 'question')
        metrics = [col for col in plot_df.columns if col != 'question']
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        width = 0.8 / len(metrics)  # Width for each metric bar

        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            positions = [j + width * i for j in range(len(plot_df))]
            values = plot_df[metric].tolist()
            plt.bar(positions, values, width=width, label=metric)

        # Set up x-axis with question labels
        question_labels = [f"Question {i+1}" for i in range(len(plot_df))]
        plt.xticks([j + width * (len(metrics) - 1) / 2 for j in range(len(plot_df))],
                   question_labels, rotation=45, ha='right')

        # Add average lines if available
        if not avg_row.empty:
            for i, metric in enumerate(metrics):
                if metric in avg_row.columns and not avg_row[metric].isna().all():
                    avg_value = avg_row[metric].iloc[0]
                    if avg_value is not None:
                        plt.axhline(y=avg_value, color=f'C{i}', linestyle='--', alpha=0.7)

        # Format and save the plot
        plt.ylabel('Score (0-1)')
        plt.title('RAGAS Evaluation Results')
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(metrics))

        # Save and close
        output_image_path = json_file_path.replace('.json', '_chart.png')
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_image_path

    except Exception as e:
        print(f"Error generating chart: {e}")
        return None
