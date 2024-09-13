import pandas as pd

file_path = r'C:\Users\hasna\Desktop\Phishing_Email1.csv'

try:
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Drop the first column
    df = df.drop(df.columns[0], axis=1)

    # Save the modified DataFrame to the same CSV file (overwriting the original file)
    df.to_csv(file_path, index=False)
    print("First column removed and file updated successfully.")
except Exception as e:
    print(f"Error processing CSV: {e}")
