import pandas as pd
from data_store import DataManager
from modify_report import choose_report_to_modify


if __name__ == '__main__':
    report_name = choose_report_to_modify()

    dm = DataManager(report_name)

    harms = dm.load_data('harms')

    harms = pd.DataFrame(harms)
    harms = harms.T

    index = harms.index.tolist() # type: ignore
    nindex = len(index) # type: ignore

    harms_df = list() # type: ignore

    for i in range(nindex):
        stakeholder = index[i] # type: ignore
        harms_list = harms.iloc[i].tolist() # type: ignore
    
        harms_df.append({'stakeholder': stakeholder, 'harms': harms_list}) # type: ignore
    
    harms_df = pd.DataFrame(harms_df)
    harms_df = harms_df.explode('harms', ignore_index=True)

    harms_df.to_excel(f'exports/{report_name.replace(' ', '_')}_harms.xlsx', index=False) # type: ignore
    print(f"Harms data exported to exports/{report_name.replace(' ', '_')}_harms.xlsx")