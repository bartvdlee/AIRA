'''
AI-generated code. Exports diverse samples of rows from a CSV file based on severity and likelihood levels.
'''

import pandas as pd
import os

def stratified_sample_by_severity_likelihood(df, n_samples=30):
    """
    Sample rows from dataframe ensuring diversity in severity and likelihood levels.
    Uses stratified sampling to get representative samples from each combination.
    """
    # Get unique combinations of severity and likelihood
    if 'Severity level' in df.columns and 'Likelihood level' in df.columns:
        # Create stratification groups
        df['strata'] = df['Severity level'].astype(str) + '_' + df['Likelihood level'].astype(str)
        unique_strata = df['strata'].unique()
        
        # Calculate samples per stratum
        samples_per_stratum = max(1, n_samples // len(unique_strata))
        remaining_samples = n_samples - (samples_per_stratum * len(unique_strata))
        
        sampled_rows = []
        
        for stratum in unique_strata:
            stratum_df = df[df['strata'] == stratum]
            n_stratum_samples = samples_per_stratum
            
            # Distribute remaining samples
            if remaining_samples > 0:
                n_stratum_samples += 1
                remaining_samples -= 1
            
            # Sample from this stratum
            if len(stratum_df) <= n_stratum_samples:
                sampled_rows.append(stratum_df)
            else:
                sampled_rows.append(stratum_df.sample(n=n_stratum_samples, random_state=42))
        
        result = pd.concat(sampled_rows, ignore_index=True)
        result = result.drop('strata', axis=1)
        
        # If we still need more samples, randomly sample the remainder
        if len(result) < n_samples:
            remaining_needed = n_samples - len(result)
            excluded_indices = result.index
            remaining_df = df[~df.index.isin(excluded_indices)]
            if len(remaining_df) > 0:
                additional_samples = remaining_df.sample(
                    n=min(remaining_needed, len(remaining_df)), 
                    random_state=42
                )
                result = pd.concat([result, additional_samples], ignore_index=True)
        
        return result
    else:
        # Fallback to random sampling if severity/likelihood columns not found
        return df.sample(n=n_samples, random_state=42)


if __name__ == '__main__':
    for report_name in os.listdir('reports'):
        df = pd.read_csv(f"reports/{report_name}/output.csv", index_col=0) # type: ignore

        sampled_df = stratified_sample_by_severity_likelihood(df)

        sampled_df.to_excel(f'exports/{report_name.replace(" ", "_")}_diverse_samples.xlsx', index=False)

        print(f"Exported {len(sampled_df)} diverse samples to exports/{report_name.replace(' ', '_')}_diverse_samples.xlsx")
        
        # Print distribution summary
        if 'Severity level' in sampled_df.columns and 'Likelihood level' in sampled_df.columns:
            print("\nSample distribution:")
            distribution = sampled_df.groupby(['Severity level', 'Likelihood level']).size()
            print(distribution)

