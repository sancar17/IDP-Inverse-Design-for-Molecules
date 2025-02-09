import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
import os
import io
from rdkit import Chem
from rdkit.Chem import SanitizeFlags


# Function to check chemical validity
def is_chemically_valid(smiles):
    try:
        # Generate a molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False  # Invalid SMILES
        
        # Perform chemical sanity checks
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
        return True  # Molecule passed all checks
    except Exception as e:
        return False  # Molecule failed sanitization


# Function to create molecule images with validity indicators
def create_molecule_image(smiles, is_valid, save_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Generate 2D coordinates
        Chem.rdDepictor.Compute2DCoords(mol)
        # Draw molecule
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.drawOptions().clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        
        # Add validity indicator
        img = Image.open(io.BytesIO(png_data))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            font = ImageFont.load_default()

        indicator = "Valid" if is_valid else "Invalid"
        color = "green" if is_valid else "red"
        draw.text((10, 10), indicator, fill=color, font=font)
        
        # Save the image
        img.save(save_path)


# Main function to process CSV files
def process_csv(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(input_csv)

    # Check for SMILES column
    if "smiles" not in data.columns:
        raise ValueError("The input CSV file must contain a 'smiles' column.")

    data["is_valid"] = data["smiles"].apply(is_chemically_valid)

    # Create images for each molecule
    for idx, row in data.iterrows():
        smiles = row["smiles"]
        is_valid = row["is_valid"]
        image_filename = f"molecule_{idx + 1}_{'valid' if is_valid else 'invalid'}.png"
        image_path = os.path.join(output_dir, image_filename)
        create_molecule_image(smiles, is_valid, image_path)

    # Save updated CSV with validity column
    updated_csv_path = os.path.join(output_dir, "updated_data.csv")
    data.to_csv(updated_csv_path, index=False)
    print(f"Processed data saved to {updated_csv_path}")


# Example usage
if __name__ == "__main__":
    input_csv = "/home/ece/Inverse-Design-For-Molecules/posterior_sampling_new/point/point_sampling_results_excluded_v3_model/homo_target_-0.200_molecules.csv"  # Path to the input CSV
    output_dir = "/home/ece/Inverse-Design-For-Molecules/posterior_sampling_new/seperate_point_sampling_validity_check"  # Directory to save the images and updated CSV
    process_csv(input_csv, output_dir)
