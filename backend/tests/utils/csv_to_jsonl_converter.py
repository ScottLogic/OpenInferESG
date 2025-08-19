import csv
import json
import os

def convert_csv_to_jsonl():
    """
    Convert the question_and_groundtruth_dataset.csv to a JSONL file format required by Ragas
    for evaluation purposes.
    """
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    input_csv_path = os.path.join(parent_dir, 'files', 'question_and_groundtruth_dataset.csv')
    output_jsonl_path = os.path.join(parent_dir, 'files', 'ragas_evaluation_dataset.jsonl')
    
    # Create the JSONL file
    with open(input_csv_path, 'r', encoding='utf-8-sig') as csv_file, \
         open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:

        # Use CSV reader with proper handling for quoted fields
        csv_reader = csv.DictReader(csv_file, skipinitialspace=True)

        # Debug: Print the available column names
        column_names = csv_reader.fieldnames
        print(f"Available columns in CSV: {column_names}")

        for row in csv_reader:
            # Create a JSON object for each row
            try:
                json_obj = {
                    "user_input": row["user_input"].strip(),
                    "reference": row["reference"].strip(),
                    "reference_contexts": [row["reference_contexts"].strip()]  # As a list as per Ragas requirement
                }

                # Write each JSON object as a line in the JSONL file
                jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            except KeyError as e:
                # Handle missing columns
                print(f"Error: Column {e} not found in the CSV. Available columns: {column_names}")
                print(f"Row data: {row}")
                raise

    print(f"Conversion completed. JSONL file created at: {output_jsonl_path}")

if __name__ == "__main__":
    convert_csv_to_jsonl()
