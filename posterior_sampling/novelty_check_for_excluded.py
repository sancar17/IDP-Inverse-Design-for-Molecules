import pandas as pd

def analyze_smiles_overlap(input_file, entire_dataset_file, excluded_dataset_file):
    """
    Analyze overlap of SMILES between input file and two reference datasets.
    
    Args:
        input_file: CSV file containing SMILES to analyze
        entire_dataset_file: CSV file containing the entire dataset
        excluded_dataset_file: CSV file containing the excluded dataset
    """
    try:
        # Read all datasets
        input_df = pd.read_csv(input_file)
        entire_df = pd.read_csv(entire_dataset_file)
        excluded_df = pd.read_csv(excluded_dataset_file)
        
        # Convert SMILES columns to sets for efficient comparison
        input_smiles = set(input_df['smiles'].unique())
        entire_smiles = set(entire_df['smiles'].unique())
        excluded_smiles = set(excluded_df['smiles'].unique())
        
        # Calculate overlaps
        in_entire = input_smiles.intersection(entire_smiles)
        in_excluded = input_smiles.intersection(excluded_smiles)
        not_found = input_smiles - (entire_smiles.union(excluded_smiles))
        
        # Print results
        print("\nAnalysis Results:")
        print("-" * 50)
        print(f"Total unique SMILES in input file: {len(input_smiles)}")
        print(f"SMILES found in entire dataset: {len(in_entire)}")
        print(f"SMILES found in excluded dataset: {len(in_excluded)}")
        print(f"SMILES not found in either dataset: {len(not_found)}")
        
        # Print percentages
        total = len(input_smiles)
        print("\nPercentages:")
        print("-" * 50)
        print(f"In entire dataset: {(len(in_entire)/total)*100:.2f}%")
        print(f"In excluded dataset: {(len(in_excluded)/total)*100:.2f}%")
        print(f"Not found in either: {(len(not_found)/total)*100:.2f}%")
        
        # Optional: Save detailed results
        results_df = pd.DataFrame({
            'SMILES': list(input_smiles),
            'In_Entire_Dataset': [smiles in entire_smiles for smiles in input_smiles],
            'In_Excluded_Dataset': [smiles in excluded_smiles for smiles in input_smiles]
        })
        results_df.to_csv('smiles_analysis_results.csv', index=False)
        print("\nDetailed results saved to 'smiles_analysis_results.csv'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
    except KeyError:
        print("Error: One or more files missing 'smiles' column")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # File paths
    input_file = "/home/ece/Inverse-Design-For-Molecules/posterior_sampling_new/intersection/intersection_sampling_results_excluded_v3_model/lumo_r2_molecules_0.000_1500.000.csv"
    entire_dataset = "/home/ece/Inverse-Design-For-Molecules/data/qm9_smiles_prop.csv"
    excluded_dataset = "/home/ece/Inverse-Design-For-Molecules/data/excluded1_qm9_smiles_prop_normalized.csv"
    
    analyze_smiles_overlap(input_file, entire_dataset, excluded_dataset)