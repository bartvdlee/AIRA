import pandas as pd
from ast import literal_eval
import os
from data_store import DataManager


if __name__ == '__main__':
    for report_name in os.listdir('reports'):
        dm = DataManager(report_name)
        info: pd.DataFrame = pd.read_csv(f"reports/{report_name}/info.csv", index_col=0) # type: ignore

        harms = dm.load_data('harms')
        human_rights = dm.load_data('human_rights')

        problematic_behaviours = literal_eval(str(info.loc['problematic_behaviours'].iloc[0])) # type: ignore

        human_rights = human_rights['human_rights']

        df_data: list[dict[str, str]] = list()

        for stakeholder in list(harms.keys()):
            for index in range(len(harms[stakeholder])):
                df_data.append({
                    'stakeholder': stakeholder.strip(),
                    'problematic_behaviour': problematic_behaviours[index].strip(),
                    'harm': harms[stakeholder][index].strip(),
                    'human_rights': human_rights[stakeholder][index],
                    'is_human_right_affected': '',
                })

        human_rights = pd.DataFrame(df_data)
        human_rights = human_rights.explode('human_rights', ignore_index=True)

        human_rights.to_excel(f'exports/{report_name.replace(' ', '_')}_human_rights.xlsx', index=False) # type: ignore
        print(f"Human rights data exported to exports/{report_name.replace(' ', '_')}_human_rights.xlsx")