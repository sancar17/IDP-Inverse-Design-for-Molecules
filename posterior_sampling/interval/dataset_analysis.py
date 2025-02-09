import pandas as pd
import numpy as np
from scipy import stats

def analyze_dataset(data_path):
    """Analyze dataset and determine meaningful intervals for each property"""
    df = pd.read_csv(data_path)
    intervals = {}
    
    properties = ['homo', 'lumo', 'r2']
    for prop in properties:
        # Get distribution info
        values = df[prop].values
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        
        # Define intervals using IQR
        low = q1 - 0.5 * iqr
        high = q3 + 0.5 * iqr
        
        intervals[prop] = {
            'intervals': [(low, q1), (q1, q3), (q3, high)],
            'stats': {
                'mean': np.mean(values),
                'std': np.std(values),
                'percentiles': [q1, q2, q3]
            }
        }
        
        '''# Save molecules in each interval
        for i, (start, end) in enumerate(intervals[prop]['intervals']):
            mask = (df[prop] >= start) & (df[prop] <= end)
            subset = df[mask]
            subset.to_csv(f'{prop}_interval_{i+1}.csv', index=False)'''
    
    return intervals