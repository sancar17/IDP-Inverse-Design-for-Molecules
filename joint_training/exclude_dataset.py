import pandas as pd

def filter_csv(input_file: str, output_file: str):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Filter rows based on conditions
    filtered_df = df[
        ~((df['lumo'] >= -0.01) & (df['lumo'] <= 0.01)) &  # Exclude rows where homo is between -0.01 and 0.01
        ~((df['r2'] >= 1300) & (df['r2'] <= 1800))          # Exclude rows where r2 is between 1300 and 1800
    ]
    
    # Write the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")

# Example usage:
input_file = '/home/ece/Inverse-Design-For-Molecules/data/qm9_smiles_prop.csv'
output_file = '/home/ece/Inverse-Design-For-Molecules/data/excluded1_qm9_smiles_prop.csv'
filter_csv(input_file, output_file)
