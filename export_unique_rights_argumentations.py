import pandas as pd
import os


if __name__ == '__main__':
    for report_name in os.listdir('reports'):
        df = pd.read_csv(f"reports/{report_name}/output.csv", index_col=0) # type: ignore

        df_human_rights = df[['Human rights impact', 'Human rights argumentation']]
        df_human_rights['Human rights impact'] = [human_right.split(' (')[0] for human_right in df_human_rights['Human rights impact']]

        df_data = list()

        for right in df_human_rights['Human rights impact'].unique():
            right_df = df_human_rights[df_human_rights['Human rights impact'] == right]
            argumentations = right_df['Human rights argumentation'].sample(n=min(5, len(right_df))).tolist()

            df_data.append({
                'Human rights impact': right,
                'Human rights argumentation': argumentations
            })
        
        df_human_rights = pd.DataFrame(df_data)
        df_human_rights = df_human_rights.explode('Human rights argumentation', ignore_index=True)
        df_human_rights.to_excel(f'exports/{report_name.replace(" ", "_")}_human_rights_argumentations.xlsx', index=False)
        print(f"Human rights argumentations data exported to exports/{report_name.replace(' ', '_')}_human_rights_argumentations.xlsx")