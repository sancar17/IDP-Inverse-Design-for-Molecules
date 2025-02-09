import pandas as pd
import numpy as np
from rdkit import Chem
from PIL import Image, ImageDraw, ImageFont
import io
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

def create_molecule_grid(molecules_data, title, save_path):
    """Create a grid of molecules with their property values"""
    n_cols = 5  # Changed to 5 columns for better layout with 10 molecules
    n_rows = (len(molecules_data) + n_cols - 1) // n_cols
    
    mol_size = (350, 350)
    img_size = (mol_size[0] * n_cols, mol_size[1] * n_rows + 60)
    
    grid = Image.new('RGB', img_size, (252, 252, 252))
    draw = ImageDraw.Draw(grid)
    
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Add main title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (img_size[0] - title_width) // 2
    draw.text((title_x, 10), title, font=title_font, fill=(50, 50, 50))
    
    def draw_molecule(mol_data, pos_x, pos_y):
        mol = Chem.MolFromSmiles(mol_data['smiles'])
        mol_copy = Chem.Mol(mol)
        try:
            mol_copy.GetConformer()
        except:
            AllChem.Compute2DCoords(mol_copy)
        
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1]-100)
        drawer.drawOptions().clearBackground = True
        drawer.drawOptions().backgroundColour = (1, 1, 1)
        drawer.drawOptions().bondLineWidth = 2
        
        drawer.DrawMolecule(mol_copy)
        drawer.FinishDrawing()
        
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        
        # Background and border
        draw.rectangle([pos_x, pos_y - 25, pos_x + mol_size[0], pos_y + mol_size[1]], 
                      fill='white', outline=(200, 200, 200), width=1)
        
        # Center molecule image
        img_x = pos_x + (mol_size[0] - img.width) // 2
        img_y = pos_y + 10
        grid.paste(img, (img_x, img_y))
        
        # Add SMILES
        max_len = 40
        display_smiles = mol_data['smiles'] if len(mol_data['smiles']) <= max_len else mol_data['smiles'][:max_len] + "..."
        smiles_bbox = draw.textbbox((0, 0), display_smiles, font=small_font)
        smiles_width = smiles_bbox[2] - smiles_bbox[0]
        smiles_x = pos_x + (mol_size[0] - smiles_width) // 2
        draw.text((smiles_x, pos_y + mol_size[1] - 135), display_smiles, font=small_font, fill=(100, 100, 100))
        
        # Add property values
        start_y = pos_y + mol_size[1] - 110
        spacing = 25
        
        properties = [
            ('HOMO', mol_data['homo']),
            ('LUMO', mol_data['lumo']),
            ('R2', mol_data['r2'])
        ]
        
        # Add reason for selection if available
        if 'reason' in mol_data:
            line = f"Type: {mol_data['reason']}"
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = pos_x + (mol_size[0] - line_width) // 2
            
            draw.rectangle([line_x - 2, 
                          start_y - spacing - 2,
                          line_x + line_width + 2,
                          start_y - spacing + line_bbox[3] - line_bbox[1] + 2],
                          fill='white')
            draw.text((line_x, start_y - spacing), line, font=font, fill=(50, 50, 50))
        
        for i, (label, value) in enumerate(properties):
            line = f"{label}: {value:.3f}"
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = pos_x + (mol_size[0] - line_width) // 2
            
            # White background for text
            text_padding = 2
            draw.rectangle([line_x - text_padding, 
                          start_y + i * spacing - text_padding,
                          line_x + line_width + text_padding,
                          start_y + i * spacing + line_bbox[3] - line_bbox[1] + text_padding],
                          fill='white')
            draw.text((line_x, start_y + i * spacing), line, font=font, fill=(50, 50, 50))
    
    # Draw all molecules
    for idx, mol_data in enumerate(molecules_data):
        row = idx // n_cols
        col = idx % n_cols
        x_offset = col * mol_size[0]
        y_offset = row * mol_size[1] + 40
        draw_molecule(mol_data, x_offset, y_offset)
    
    return grid

def find_representative_molecules(data_path):
    """Find molecules with mean values and edge cases for each property"""
    # Read data
    df = pd.read_csv(data_path)
    
    # Calculate means and standard deviations
    means = {
        'homo': df['homo'].mean(),
        'lumo': df['lumo'].mean(),
        'r2': df['r2'].mean()
    }
    
    stds = {
        'homo': df['homo'].std(),
        'lumo': df['lumo'].std(),
        'r2': df['r2'].std()
    }
    
    # Function to calculate distance from mean for all properties
    def distance_from_mean(row):
        return np.sqrt(
            ((row['homo'] - means['homo']) / stds['homo']) ** 2 +
            ((row['lumo'] - means['lumo']) / stds['lumo']) ** 2 +
            ((row['r2'] - means['r2']) / stds['r2']) ** 2
        )
    
    # Find molecules close to mean for all properties
    df['mean_distance'] = df.apply(distance_from_mean, axis=1)
    mean_molecules = df.nsmallest(10, 'mean_distance').copy()
    mean_molecules['reason'] = 'Average'
    
    # Find edge cases for each property (both max and min)
    edge_cases = []
    for prop in ['homo', 'lumo', 'r2']:
        # Find max values
        max_mol = df.loc[df[prop].idxmax()].copy()
        max_mol['reason'] = f'Max {prop.upper()}'
        edge_cases.append(max_mol)
        
        # Find min values
        min_mol = df.loc[df[prop].idxmin()].copy()
        min_mol['reason'] = f'Min {prop.upper()}'
        edge_cases.append(min_mol)
        
        # Find most extreme relative to mean (outliers)
        df[f'{prop}_zscore'] = (df[prop] - means[prop]) / stds[prop]
        outlier = df.loc[df[f'{prop}_zscore'].abs().idxmax()].copy()
        outlier['reason'] = f'Extreme {prop.upper()}'
        edge_cases.append(outlier)
    
    # Take the 10 most unique edge cases (some might be duplicates)
    edge_df = pd.DataFrame(edge_cases).drop_duplicates(subset='smiles').head(10)
    
    # Combine results
    all_molecules = pd.concat([mean_molecules, edge_df])
    
    return all_molecules

def main(data_path, output_path):
    # Find representative molecules
    molecules = find_representative_molecules(data_path)
    
    # Create visualization
    molecules_list = molecules.to_dict('records')
    
    # Create two separate visualizations
    mean_mols = [m for m in molecules_list if m['reason'] == 'Average']
    edge_mols = [m for m in molecules_list if m['reason'] != 'Average']
    
    # Generate and save visualizations
    mean_grid = create_molecule_grid(mean_mols, "10 Molecules with Average Properties", 
                                   f"{output_path}_mean.png")
    edge_grid = create_molecule_grid(edge_mols, "10 Edge Case Molecules", 
                                   f"{output_path}_edge.png")
    
    mean_grid.save(f"{output_path}_mean.png")
    edge_grid.save(f"{output_path}_edge.png")
    
    # Print summary
    print("\nMolecules with average properties:")
    for mol in mean_mols:
        print(f"\nSMILES: {mol['smiles']}")
        print(f"HOMO: {mol['homo']:.3f}")
        print(f"LUMO: {mol['lumo']:.3f}")
        print(f"R2: {mol['r2']:.3f}")
    
    print("\nEdge case molecules:")
    for mol in edge_mols:
        print(f"\nSMILES: {mol['smiles']}")
        print(f"Type: {mol['reason']}")
        print(f"HOMO: {mol['homo']:.3f}")
        print(f"LUMO: {mol['lumo']:.3f}")
        print(f"R2: {mol['r2']:.3f}")

if __name__ == "__main__":
    data_path = "../../data/qm9_smiles_prop_normalized.csv"
    output_path = "representative_molecules"
    main(data_path, output_path)