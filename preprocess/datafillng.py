## Author: Neel Kanwal, neel.kanwal0@gmail.com
# This was used as preprocessing to fill mising values in the excel sheet like Suction, Stimulation, and Resus.

import pandas as pd
import numpy as np

# Load the Excel sheet into a DataFrame
df = pd.read_excel("D:\\Moyo\\matched_details.xlsx")

# RULE 0: IF all four columns are empty
num_rows_all_nan = (df[['Resuscitation', 'Stimulation', 'BMV', 'Suction']].isna().sum(axis=1) == 4).sum()
print(f"Before: Number of rows where Resuscitation, Stimulation, BMV, and Suction are all NaN: {num_rows_all_nan}")
df['Notes'] = np.nan
mask = (df[['Resuscitation', 'Stimulation', 'BMV', 'Suction']].isna().sum(axis=1) == 4)
# Fill the four fields with 2 where the mask is True
df.loc[mask, ['Resuscitation', 'Stimulation', 'BMV', 'Suction']] = 0
# df.loc[mask, 'Notes'] = "RES, SIM, SUC and BMV are set to 2;" + df.loc[mask, 'Notes']
df.loc[mask, 'Notes'] = "RES, SIM, SUC and BMV are set to 0;"
num_rows_all_nan = (df[['Resuscitation', 'Stimulation', 'BMV', 'Suction']].isna().sum(axis=1) == 4).sum()
print(f"After: Number of rows where Resuscitation, Stimulation, BMV, and Suction are all NaN: {num_rows_all_nan}")

#RULE 01 : IF three columns ( 'Stimulation', 'BMV', 'Suction') are empty but Resuscitation value exists
num_rows_all_nan = (df[['BMV', 'Suction', 'Stimulation']].isna().sum(axis=1) == 3).sum()
print(f"Before: Number of rows where Stimulation, BMV, and Suction are all NaN: {num_rows_all_nan}")

mask_resus_2 = df['Resuscitation'] == 2
cols_to_check = ['BMV', 'Suction', 'Stimulation']
for col in cols_to_check:
    mask_col = df[col].isna()
    df.loc[mask_resus_2 & mask_col, col] = 2
    df.loc[mask_resus_2 & mask_col, 'Notes'] = df.loc[mask_resus_2 & mask_col, 'Notes'].fillna('') + f" {col} is set to 2"

mask_resus_1 = df['Resuscitation'] == 1
cols_to_check = ['BMV', 'Suction', 'Stimulation']
for col in cols_to_check:
    mask_col = df[col].isna()
    df.loc[mask_resus_1 & mask_col, col] = 0
    df.loc[mask_resus_1 & mask_col, 'Notes'] = df.loc[mask_resus_1 & mask_col, 'Notes'].fillna('') + f" {col} is set to 0"

num_rows_all_nan = (df[['BMV', 'Suction', 'Stimulation']].isna().sum(axis=1) == 3).sum()
print(f"After: Number of rows where 'BMV', and 'Suction' were filled with 2: {num_rows_all_nan}")

#RULE 02 : IF values in one or all three columns ( 'Stimulation', 'BMV', 'Suction') exists but Resuscitation is empty.
mask_resus_nan_and_one = (df['Resuscitation'].isna()) & (df[['Stimulation', 'BMV', 'Suction']].eq(1).any(axis=1))
# Fill Resuscitation with 1 for those rows
df.loc[mask_resus_nan_and_one, 'Resuscitation'] = 1
df.loc[mask_resus_nan_and_one, 'Notes'] = df.loc[mask_resus_nan_and_one, 'Notes'].fillna('') + f" RESUC is set to 1"

mask_resus_nan_and_all_two = (df['Resuscitation'].isna()) & (df[['Stimulation', 'BMV', 'Suction']].eq(2).all(axis=1))
df.loc[mask_resus_nan_and_all_two, 'Resuscitation'] = 2
df.loc[mask_resus_nan_and_all_two, 'Notes'] = df.loc[mask_resus_nan_and_all_two, 'Notes'].fillna('') + " Resuscitation is set to 2 based on Rule 02"

# Fill missing values with 0 in the three columns for those rows
for col in ['Stimulation', 'BMV', 'Suction']:
    mask_col = df[col].isna()
    df.loc[mask_resus_nan_and_one, col] = df.loc[mask_resus_nan_and_one, col].fillna(0)
    df.loc[mask_resus_nan_and_one, 'Resuscitation'] = df.loc[mask_resus_nan_and_one, 'Resuscitation'].fillna(0)
    df.loc[mask_resus_nan_and_one & mask_col, 'Notes'] = df.loc[mask_resus_nan_and_one & mask_col, 'Notes'].fillna(
        '') + f" {col} is set to 0"

# RULE 03: IF Outcome_24 hours is empty, fill with Outcome_30mins
num_rows_nan = df['Outcome_24hours'].isna().sum()
print(f"Before: Number of rows where 'Outcome_24hours' is NaN: {num_rows_nan}")
mask = df['Outcome_24hours'].isna()
df.loc[mask, 'Outcome_24hours'] = df.loc[mask, 'Outcome_30min']
df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " Outcome_24hours is updated with Outcome_30min;"
num_rows_nan = df['Outcome_24hours'].isna().sum()
print(f"After: Number of rows where 'Outcome_24hours' is NaN: {num_rows_nan}")

# RULE 04: Create new column for Outcome_30mins change (seisures) 6-->2, and 1-->1, 2-->2 and anything else to 3
mask = (~df['Outcome_30min'].isin([1, 2, 3]))
num_rows_not_123 = mask.sum()
print(f"Before: Number of rows in 'Outcome_30min' where it is not 1, 2, or 3: {num_rows_not_123}")
df['nOutcome_30min'] = df['Outcome_30min']
# Replace values in 'nOutcome_30min' based on the given rules
df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') +  " nOutcome_30min with rule to make three classes;"
df.loc[df['nOutcome_30min'] == 6, 'nOutcome_30min'] = 2
df.loc[df['nOutcome_30min'] >= 3, 'nOutcome_30min'] = 3
mask = ~df['nOutcome_30min'].isin([1, 2, 3])
num_rows_not_123 = mask.sum()
print(f"After: Number of rows in 'nOutcome_30min' where it is not 1, 2, or 3: {num_rows_not_123}")

# RULE 05: Create new column for Outcome_24hours with rule 1-->1, 2-->2 and anything else to 3
df['nOutcome_24hours'] = df['Outcome_24hours']
mask = (~df['Outcome_24hours'].isin([1, 2, 3]))
num_rows_not_123 = mask.sum()
print(f"Before: Number of rows in 'Outcome_24hours' where it is not 1, 2, or 3: {num_rows_not_123}")
# Replace values in 'nOutcome_30min' based on the given rules
df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " nOutcome_24hours with rule to make three classes;"
df.loc[df['nOutcome_24hours'] == 6, 'nOutcome_24hours'] = 2
df.loc[df['nOutcome_24hours'] >= 3, 'nOutcome_24hours'] = 3
mask = ~df['nOutcome_24hours'].isin([1, 2, 3])
num_rows_not_123 = mask.sum()
print(f"After: Number of rows in 'nOutcome_24hours' where it is not 1, 2, or 3: {num_rows_not_123}")


columns_order = ['filename', 'Apgar_5min', 'Resuscitation', 'Stimulation', 'Suction', 'BMV',
                 'Outcome_30min', 'nOutcome_30min', 'Outcome_24hours', 'nOutcome_24hours', 'Notes']
df = df[columns_order]

df.to_excel('D:\\Moyo\\matched_details-modified.xlsx', index=False)


### OLD RULES.
## RULE 01: IF all four columns are empty
# num_rows_all_nan = (df[['Resuscitation', 'Stimulation', 'BMV', 'Suction']].isna().sum(axis=1) == 4).sum()
# print(f"Before: Number of rows where Resuscitation, Stimulation, BMV, and Suction are all NaN: {num_rows_all_nan}")
# mask = (df[['Resuscitation', 'Stimulation', 'BMV', 'Suction']].isna().sum(axis=1) == 4)
# # Fill the four fields with 2 where the mask is True
# df.loc[mask, ['Resuscitation', 'Stimulation', 'BMV', 'Suction']] = 2
# # df.loc[mask, 'Notes'] = "RES, SIM, SUC and BMV are set to 2;" + df.loc[mask, 'Notes']
# df.loc[mask, 'Notes'] = "RES, SIM, SUC and BMV are set to 2;"
# num_rows_all_nan = (df[['Resuscitation', 'Stimulation', 'BMV', 'Suction']].isna().sum(axis=1) == 4).sum()
# print(f"After: Number of rows where Resuscitation, Stimulation, BMV, and Suction are all NaN: {num_rows_all_nan}")
#
# # RULE 02 : IF all three columns are empty
# num_rows_all_nan = (df[['Stimulation', 'BMV', 'Suction']].isna().sum(axis=1) == 3).sum()
# print(f"Before: Number of rows where Stimulation, BMV, and Suction are all NaN: {num_rows_all_nan}")
# mask = ((df[['Stimulation', 'BMV', 'Suction']].isna().sum(axis=1) == 3) & (~df['Resuscitation'].isna()))
# df.loc[mask, ['Stimulation', 'BMV', 'Suction']] = 2
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " SIM, SUC and BMV are set to 2;"
# num_rows_all_nan = (df[['Stimulation', 'BMV', 'Suction']].isna().sum(axis=1) == 3).sum()
# print(f"After: Number of rows where 'Stimulation', 'BMV', and 'Suction' are all NaN: {num_rows_all_nan}")
#
# # RULE 03: If there are values in three columns but RESUS column is empty, then fill it with 1 if any of those column is one.
# mask = ((df[['Stimulation', 'BMV', 'Suction']] == 1).any(axis=1) & df['Resuscitation'].isna())
# num_cols_with_one = mask.sum()
# print(f"Before: Number of rows where either of Stimulation, BMV, and Suction has 1 value: {num_cols_with_one}")
# df.loc[mask, 'Resuscitation'] = 1
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " RESUC is set to 1 as one of 'Stimulation', 'BMV', or 'Suction' is 1;"
# # df.loc[~mask, 'Resuscitation'] = df['Resuscitation'].fillna(2)
# # df.loc[~mask, 'Notes'] = df.loc[~mask, 'Notes'].fillna('') + " RESUC is set to 2 as NONE of 'Stimulation', 'BMV', or 'Suction' is 1;"
# mask = ((df[['Stimulation', 'BMV', 'Suction']] == 1).any(axis=1) & df['Resuscitation'].isna())
# num_cols_with_one = mask.sum()
# print(f"After: Number of rows where either of Stimulation, BMV, and Suction has 1 value: {num_cols_with_one}")
#
#
# # RULE 04 : IF all two columns ('BMV', 'Suction') are empty and RESUC is not applied
# num_rows_all_nan = (df[['BMV', 'Suction']].isna().sum(axis=1) == 2).sum()
# print(f"Before: Number of rows where BMV, and Suction are all NaN: {num_rows_all_nan}")
# # mask = ((df[['BMV', 'Suction']].isna().sum(axis=1) == 2) & (df['Resuscitation'] == 2))
# mask = (df[['BMV', 'Suction']].isna().sum(axis=1) == 2)
# df.loc[mask, ['BMV', 'Suction']] = 2
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " SUC and BMV are set to 2"
# num_rows_all_nan = (df[['BMV', 'Suction']].isna().sum(axis=1) == 2).sum()
# print(f"After: Number of rows where 'BMV', and 'Suction' were filled with 2: {num_rows_all_nan}")
#
# # RULE 05: IF either 'BMV' or 'SUC' is empty
# mask = (df[['BMV', 'Suction']].isna().any(axis=1))
# num_rows = mask.sum()
# print(f"Before: Number of rows where 'BMV' or 'Suction' is NaN: {num_rows}")
# mask = (df[['BMV']].isna().sum(axis=1) == 1)
# df.loc[mask, 'BMV'] = 2
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " BMV is set to 2;"
# mask = (df[['Suction']].isna().sum(axis=1) == 1)
# df.loc[mask, 'Suction'] = 2
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " SUC is set to 2;"
# mask = df[['BMV', 'Suction']].isna().any(axis=1)
# num_rows = mask.sum()
# print(f"After: Number of rows where 'BMV' or 'Suction' is NaN: {num_rows}")
#
# # RULE 06: IF 'Stimulation' is empty
# num_rows_all_nan = (df[['Stimulation']].isna().sum(axis=1) == 1).sum()
# print(f"Before: Number of rows where Stimulation is NaN: {num_rows_all_nan}")
# mask = ((df[['Stimulation']].isna().sum(axis=1) == 1) & (df['Resuscitation'] == 1)) # where RESUC is 1
# df.loc[mask, 'Stimulation'] = 1
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " Stimulation is set to 1;"
# mask = ((df[['Stimulation']].isna().sum(axis=1) == 1) & (df['Resuscitation'] == 2)) # where RESUC is 2
# df.loc[mask, 'Stimulation'] = 2 # where RESUC is 2
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " Stimulation is set to 2;"
# num_rows_all_nan = (df[['Stimulation']].isna().sum(axis=1) == 1).sum()
# print(f"After: Number of rows where Stimulation is NaN: {num_rows_all_nan}")
#
# # RULE 07: IF 'SUC' is 3 (not penguin) then make it 1 (penguin)
# num_rows_suction_3 = (df['Suction'] == 3).sum()
# print(f"Before: Number of rows where 'Suction' is 3: {num_rows_suction_3}")
# mask = (df['Suction'] == 3)
# df.loc[mask, 'Suction'] = 1 # df.loc[df['Suction'] == 3, 'Suction'] = 1
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " SUC value changed from 3 to 1;"
# num_rows_suction_3 = (df['Suction'] == 3).sum()
# print(f"After:Number of rows where 'Suction' is 3: {num_rows_suction_3}")
#
# # RULE 08: IF Outcome_24 hours is empty, fill with Outcome_30mins
# num_rows_nan = df['Outcome_24hours'].isna().sum()
# print(f"Before: Number of rows where 'Outcome_24hours' is NaN: {num_rows_nan}")
# mask = df['Outcome_24hours'].isna()
# df.loc[mask, 'Outcome_24hours'] = df.loc[mask, 'Outcome_30min']
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " Outcome_24hours is updated with Outcome_30min;"
# num_rows_nan = df['Outcome_24hours'].isna().sum()
# print(f"After: Number of rows where 'Outcome_24hours' is NaN: {num_rows_nan}")
#
# # RULE 09: Create new column for Outcome_30mins change (seisures) 6-->2, and 1-->1, 2-->2 and anything else to 3
# mask = (~df['Outcome_30min'].isin([1, 2, 3]))
# num_rows_not_123 = mask.sum()
# print(f"Before: Number of rows in 'Outcome_30min' where it is not 1, 2, or 3: {num_rows_not_123}")
# df['nOutcome_30min'] = df['Outcome_30min']
# # Replace values in 'nOutcome_30min' based on the given rules
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') +  " nOutcome_30min with rule to make three classes;"
# df.loc[df['nOutcome_30min'] == 6, 'nOutcome_30min'] = 2
# df.loc[df['nOutcome_30min'] >= 3, 'nOutcome_30min'] = 3
# mask = ~df['nOutcome_30min'].isin([1, 2, 3])
# num_rows_not_123 = mask.sum()
# print(f"After: Number of rows in 'nOutcome_30min' where it is not 1, 2, or 3: {num_rows_not_123}")
#
# # RULE 10: Create new column for Outcome_24hours with rule 1-->1, 2-->2 and anything else to 3
# df['nOutcome_24hours'] = df['Outcome_24hours']
# mask = (~df['Outcome_24hours'].isin([1, 2, 3]))
# num_rows_not_123 = mask.sum()
# print(f"Before: Number of rows in 'Outcome_24hours' where it is not 1, 2, or 3: {num_rows_not_123}")
# # Replace values in 'nOutcome_30min' based on the given rules
# df.loc[mask, 'Notes'] = df.loc[mask, 'Notes'].fillna('') + " nOutcome_24hours with rule to make three classes;"
# df.loc[df['nOutcome_24hours'] == 6, 'nOutcome_24hours'] = 2
# df.loc[df['nOutcome_24hours'] >= 3, 'nOutcome_24hours'] = 3
# mask = ~df['nOutcome_24hours'].isin([1, 2, 3])
# num_rows_not_123 = mask.sum()
# print(f"After: Number of rows in 'nOutcome_24hours' where it is not 1, 2, or 3: {num_rows_not_123}")