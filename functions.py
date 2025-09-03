import pandas as pd
import numpy as np
import re

def process_demographics(data, dq3_map, dq4_map, dq6_map, dq7_map, column_config):
    """
    Processes demographic data to calculate and format percentage distributions.
    """
    t14_names = column_config['t14']
    t15_names = column_config['t15']
    t16_names = column_config['t16']
    t17_names = column_config['t17']

    demographics = data[['dq3', 'dq4', 'dq6', 'dq7']].copy()
    demographics.dropna(subset=['dq3', 'dq4', 'dq6', 'dq7'], how='all', inplace=True)
    base_dq3 = demographics['dq3'].count()
    base_dq4 = demographics['dq4'].count()
    base_dq6 = demographics['dq6'].count()
    base_dq7 = demographics['dq7'].count()
    age_map = dict(zip(dq7_map['age'], dq7_map['age_brackets']))
    demographics['dq7'] = demographics['dq7'].map(age_map)
    def calculate_distribution(df, column_name):
        if df[column_name].dropna().empty: return pd.DataFrame(columns=[column_name, t14_names['col2']])
        dist = df[column_name].dropna().value_counts(normalize=True).mul(100).reset_index()
        dist.columns = [column_name, t14_names['col2']]
        return dist
    t14 = calculate_distribution(demographics, 'dq3')
    t15 = calculate_distribution(demographics, 'dq4')
    t16 = calculate_distribution(demographics, 'dq6')
    t17 = calculate_distribution(demographics, 'dq7')
    if not t14.empty: t14['dq3'], t14[t14_names['col2']] = t14['dq3'].astype(int), t14[t14_names['col2']].round(2)
    if not t15.empty: t15['dq4'], t15[t15_names['col2']] = t15['dq4'].astype(int), t15[t15_names['col2']].round(2)
    if not t16.empty: t16['dq6'], t16[t16_names['col2']] = t16['dq6'].astype(int), t16[t16_names['col2']].round(2)
    if not t17.empty: t17[t17_names['col2']] = t17[t17_names['col2']].round(2)
    dq3_dict, dq4_dict, dq6_dict = dict(zip(dq3_map['codes'], dq3_map['values'])), dict(zip(dq4_map['codes'], dq4_map['values'])), dict(zip(dq6_map['codes'], dq6_map['values']))
    if not t14.empty: t14['dq3'] = t14['dq3'].map(dq3_dict)
    if not t15.empty: t15['dq4'] = t15['dq4'].map(dq4_dict)
    if not t16.empty: t16['dq6'] = t16['dq6'].map(dq6_dict)
    t14 = pd.concat([pd.DataFrame([{ 'dq3': 'Base', t14_names['col2']: base_dq3}]), t14], ignore_index=True)
    t15 = pd.concat([pd.DataFrame([{ 'dq4': 'Base', t15_names['col2']: base_dq4}]), t15], ignore_index=True)
    t16 = pd.concat([pd.DataFrame([{ 'dq6': 'Base', t16_names['col2']: base_dq6}]), t16], ignore_index=True)
    t17 = pd.concat([pd.DataFrame([{ 'dq7': 'Base', t17_names['col2']: base_dq7}]), t17], ignore_index=True)
    formatter = lambda x: f'{int(x)}' if pd.notna(x) and x == int(x) else f'{x:.2f}' if pd.notna(x) else ''
    for df, col_name in [(t14, t14_names['col2']), (t15, t15_names['col2']), (t16, t16_names['col2']), (t17, t17_names['col2'])]: df[col_name] = df[col_name].apply(formatter)
    t14.rename(columns={'dq3': t14_names['col1']}, inplace=True); t15.rename(columns={'dq4': t15_names['col1']}, inplace=True)
    t16.rename(columns={'dq6': t16_names['col1']}, inplace=True); t17.rename(columns={'dq7': t17_names['col1']}, inplace=True)
    return t14, t15, t16, t17

def generate_dealership_summary_tables(data, q5_map, q6a_map, column_config):
    t7_names = column_config['t7']
    t8_names = column_config['t8']
    dealership_visit_df = data[['q5', 'q6a']].copy(); dealership_visit_df.dropna(subset=['q5'], inplace=True)
    def format_values(value):
        if pd.notna(value):
            if isinstance(value, (int, float)) and value == int(value): return f'{int(value)}'
            elif isinstance(value, (int, float)): return f'{value:.2f}'
        return value
    if dealership_visit_df.empty: t7 = pd.DataFrame(columns=[t7_names['col1'], t7_names['col2']])
    else:
        q5_base_count = len(dealership_visit_df); t7 = dealership_visit_df['q5'].value_counts(normalize=True).mul(100).reset_index(); t7.columns = [t7_names['col1'], t7_names['col2']]
        q5_dict = dict(zip(q5_map['codes'], q5_map['values'])); t7[t7_names['col1']] = t7[t7_names['col1']].map(q5_dict)
        t7 = pd.concat([pd.DataFrame([{t7_names['col1']: 'Base', t7_names['col2']: q5_base_count}]), t7], ignore_index=True)
        t7[t7_names['col2']] = t7[t7_names['col2']].apply(format_values)
    q6a_filtered_df = dealership_visit_df[dealership_visit_df['q5'] == 1].copy()
    if q6a_filtered_df.empty: t8 = pd.DataFrame(columns=[t8_names['col1'], t8_names['col2']])
    else:
        q6a_base_count = len(q6a_filtered_df); t8 = q6a_filtered_df['q6a'].value_counts(normalize=True).mul(100).reset_index(); t8.columns = [t8_names['col1'], t8_names['col2']]
        q6a_dict = dict(zip(q6a_map['codes'], q6a_map['values'])); t8[t8_names['col1']] = t8[t8_names['col1']].map(q6a_dict)
        t8 = pd.concat([pd.DataFrame([{t8_names['col1']: 'Base', t8_names['col2']: q6a_base_count}]), t8], ignore_index=True)
        t8[t8_names['col2']] = t8[t8_names['col2']].apply(format_values)
    return t7, t8

def t1_create_brand_summary_from_codes(brands, aq3_po_map, column_config):
    t1_names = column_config['t1']
    t1 = pd.DataFrame(brands["aq3_po"]).copy(); mapping_dict = dict(zip(aq3_po_map['codes'], aq3_po_map['brands'])); t1['aq3_po'] = t1['aq3_po'].map(mapping_dict)
    clean_brands = t1['aq3_po'].dropna().copy()
    if clean_brands.empty: return pd.DataFrame(columns=[t1_names['col1'], t1_names['col2']])
    base_count = len(clean_brands); brand_distribution = clean_brands.value_counts(normalize=True).mul(100).reset_index(); brand_distribution.columns = [t1_names['col1'], t1_names['col2']]
    brand_distribution = brand_distribution[brand_distribution[t1_names['col2']] > 0].copy()
    final_table = pd.concat([pd.DataFrame([{t1_names['col1']: 'Base', t1_names['col2']: base_count}]), brand_distribution], ignore_index=True)
    def format_values(value):
        if pd.notna(value):
            if isinstance(value, (int, float)) and value == int(value): return f'{int(value)}'
            elif isinstance(value, (int, float)): return f'{value:.2f}'
        return value
    final_table[t1_names['col2']] = final_table[t1_names['col2']].apply(format_values)
    return final_table

def create_bike_summary_t2(data, aq3_po_map, column_config):
    t2_names = column_config['t2']
    clean_data = data['aq3_po'].dropna()
    if clean_data.empty: return pd.DataFrame(columns=[t2_names['col1'], t2_names['col2']])
    base_count = len(clean_data); distribution = clean_data.value_counts(normalize=True).mul(100).reset_index(); distribution.columns = ['codes', t2_names['col2']]
    distribution = distribution[distribution[t2_names['col2']] > 0].copy()
    bike_name_mapper = dict(zip(aq3_po_map['codes'], aq3_po_map['bike_names'])); brand_mapper = dict(zip(aq3_po_map['codes'], aq3_po_map['brands']))
    distribution[t2_names['col1']], distribution['Brand'] = distribution['codes'].map(bike_name_mapper), distribution['codes'].map(brand_mapper)
    brand_totals = distribution.groupby('Brand')[t2_names['col2']].sum().sort_values(ascending=False); brand_order = brand_totals.index.tolist()
    distribution['Brand'] = pd.Categorical(distribution['Brand'], categories=brand_order, ordered=True); distribution.sort_values(by=['Brand', t2_names['col2']], ascending=[True, False], inplace=True)
    t2 = distribution[[t2_names['col1'], 'Brand', t2_names['col2']]]
    final_table = pd.concat([pd.DataFrame([{t2_names['col1']: 'Base', 'Brand': '', t2_names['col2']: base_count}]), t2], ignore_index=True)
    def format_values(value):
        if pd.notna(value):
            if isinstance(value, (int, float)) and value == int(value): return f'{int(value)}'
            elif isinstance(value, (int, float)): return f'{value:.2f}'
        return value
    final_table[t2_names['col2']] = final_table[t2_names['col2']].apply(format_values)
    return final_table.drop(columns=['Brand'])

def t3_calculate_brand_frequency(source_df, map_df, column_config):
    t3_names = column_config['t3']
    processed_df = source_df.filter(regex=r'^aq5a_\d').copy(); processed_df.dropna(how='all', inplace=True)
    rename_mapper = dict(zip(map_df['codes'], map_df['brands'])); processed_df.rename(columns=rename_mapper, inplace=True)
    df_combined = processed_df.T.groupby(level=0).sum().T; non_zero_counts = (df_combined != 0).sum()
    result_df = non_zero_counts.reset_index(); result_df.columns = [t3_names['col1'], t3_names['col2']]
    total_respondents = len(df_combined)
    if total_respondents == 0: return pd.DataFrame([{t3_names['col1']: 'Base', t3_names['col2']: '0'}])
    result_df[t3_names['col2']] = (result_df[t3_names['col2']] / total_respondents) * 100; result_df.sort_values(by=t3_names['col2'], ascending=False, inplace=True)
    result_df[t3_names['col2']] = result_df[t3_names['col2']].map('{:.2f}'.format)
    return pd.concat([pd.DataFrame([{t3_names['col1']: 'Base', t3_names['col2']: f'{total_respondents}'}]), result_df], ignore_index=True)

def t4_calculate_bike_frequency(source_df, map_df, column_config):
    t4_names = column_config['t4']
    processed_df = source_df.filter(regex=r'^aq5a_\d').copy(); processed_df.dropna(how='all', inplace=True)
    rename_mapper = dict(zip(map_df['codes'], map_df['bike_names'])); processed_df.rename(columns=rename_mapper, inplace=True)
    non_zero_counts = (processed_df != 0).sum(); result_df = non_zero_counts.reset_index(); result_df.columns = [t4_names['col1'], t4_names['col2']]
    total_respondents = len(processed_df)
    if total_respondents == 0: return pd.DataFrame([{t4_names['col1']: 'Base', t4_names['col2']: '0'}])
    result_df[t4_names['col2']] = (result_df[t4_names['col2']] / total_respondents) * 100
    result_df = result_df[result_df[t4_names['col2']] > 0].copy()
    brand_mapper = dict(zip(map_df['bike_names'], map_df['brands'])); result_df['Brand'] = result_df[t4_names['col1']].map(brand_mapper)
    brand_totals = result_df.groupby('Brand')[t4_names['col2']].sum().sort_values(ascending=False); brand_order = brand_totals.index.tolist()
    result_df['Brand'] = pd.Categorical(result_df['Brand'], categories=brand_order, ordered=True); result_df.sort_values(by=['Brand', t4_names['col2']], ascending=[True, False], inplace=True)
    result_df = result_df[[t4_names['col1'], 'Brand', t4_names['col2']]]; result_df[t4_names['col2']] = result_df[t4_names['col2']].map('{:.2f}'.format)
    final_df_with_base = pd.concat([pd.DataFrame([{t4_names['col1']: 'Base', 'Brand': '', t4_names['col2']: f'{total_respondents}'}]), result_df], ignore_index=True)
    return final_df_with_base.drop(columns=['Brand'])

def t5_calculate_distribution(source_df, column_name, map_df, column_config):
    t5_names = column_config['t5']
    data_series = source_df[column_name]; base_count = data_series.count()
    if base_count == 0: return pd.DataFrame([{t5_names['col1']: 'Base', t5_names['col2']: '0'}])
    distribution = data_series.value_counts(normalize=True) * 100; dist_df = distribution.reset_index(); dist_df.columns = ['Code', t5_names['col2']]
    mapper = dict(zip(map_df['codes'], map_df['bike_names'])); dist_df[t5_names['col1']] = dist_df['Code'].map(mapper)
    final_table = dist_df[[t5_names['col1'], t5_names['col2']]]; final_table = pd.concat([pd.DataFrame([{t5_names['col1']: 'Base', t5_names['col2']: base_count}]), final_table], ignore_index=True)
    def format_values(value):
        if isinstance(value, (int, float)):
            if value == int(value): return f'{int(value)}'
            else: return f'{value:.2f}'
        return value
    final_table[t5_names['col2']] = final_table[t5_names['col2']].apply(format_values)
    return final_table

def t6_create_bike_summary_table(source_df, map_df, column_config):
    t6_names = column_config['t6']
    t6_df = source_df.filter(regex=r'^aq6_\d').copy()
    rename_mapper = dict(zip(map_df['codes'], map_df['bike_names'])); t6_df.rename(columns=rename_mapper, inplace=True)
    brand_mapper = dict(zip(map_df['bike_names'], map_df['brands'])); summary_list = []
    for bike_name in t6_df.columns:
        column_data = t6_df[bike_name]; base = column_data.notna().sum()
        percent_yes, percent_no = (((column_data == 2).sum() / base) * 100, ((column_data == 1).sum() / base) * 100) if base > 0 else (0.0, 0.0)
        summary_list.append({t6_names['col1']: bike_name, 'Brand': brand_mapper.get(bike_name), t6_names['col2']: int(base), t6_names['col3']: percent_yes, t6_names['col4']: percent_no})
    if not summary_list: return pd.DataFrame(columns=[t6_names['col1'], t6_names['col2'], t6_names['col3'], t6_names['col4']])
    summary_df = pd.DataFrame(summary_list); summary_df = summary_df[summary_df[t6_names['col2']] > 0].copy()
    if summary_df.empty: return summary_df.drop(columns=['Brand'], errors='ignore')
    brand_totals = summary_df.groupby('Brand')[t6_names['col2']].sum().sort_values(ascending=False); brand_order = brand_totals.index.tolist()
    summary_df['Brand'] = pd.Categorical(summary_df['Brand'], categories=brand_order, ordered=True); summary_df.sort_values(by=['Brand', t6_names['col2']], ascending=[True, False], inplace=True)
    return summary_df.drop(columns=['Brand'])

def t9_create_usage_distribution_table(source_df, column_name, map_df, column_config):
    t9_names = column_config['t9']
    data_series = source_df[column_name]; base_count = data_series.count()
    if base_count == 0: return pd.DataFrame([{t9_names['col1']: 'Base', t9_names['col2']: '0'}])
    distribution = data_series.value_counts(normalize=True) * 100; dist_df = distribution.reset_index(); dist_df.columns = ['Code', t9_names['col2']]
    mapper = dict(zip(map_df['codes'], map_df['values'])); dist_df[t9_names['col1']] = dist_df['Code'].map(mapper)
    final_table = dist_df[[t9_names['col1'], t9_names['col2']]]; final_table = pd.concat([pd.DataFrame([{t9_names['col1']: 'Base', t9_names['col2']: base_count}]), final_table], ignore_index=True)
    def format_values(value):
        if isinstance(value, (int, float)):
            if value == int(value): return f'{int(value)}'
            else: return f'{value:.2f}'
        return value
    final_table[t9_names['col2']] = final_table[t9_names['col2']].apply(format_values)
    return final_table

def t10_t12_calculate_purchase_type_distribution(source_df, map_df, column_config):
    t10_names = column_config['t10']
    t10 = source_df[['dq1a', 'dq1b']]; base = t10['dq1a'].isin([1, 2, 3]).sum()
    if base == 0: return pd.DataFrame([{t10_names['col1']: 'Base', t10_names['col2']: '0'}])
    distribution = (t10['dq1b'].value_counts() / base) * 100; dist_df = distribution.reset_index(); dist_df.columns = ['Code', t10_names['col2']]
    mapper = dict(zip(map_df['codes'], map_df['values'])); dist_df[t10_names['col1']] = dist_df['Code'].map(mapper)
    final_table = dist_df[[t10_names['col1'], t10_names['col2']]].dropna(subset=[t10_names['col1']])
    final_table = pd.concat([pd.DataFrame([{t10_names['col1']: 'Base', t10_names['col2']: base}]), final_table], ignore_index=True)
    def format_values(value):
        if isinstance(value, (int, float)):
            if value == int(value): return f'{int(value)}'
            else: return f'{value:.2f}'
        return value
    final_table[t10_names['col2']] = final_table[t10_names['col2']].apply(format_values)
    return final_table

def t13_calculate_addition_replacement(source_df, map_df, column_config):
    t13_names = column_config['t13']
    try:
        relevant_cols = ['dq2b'] + source_df.filter(regex=r'^dq2a_\d+').columns.tolist() + ['dq1b']; t13 = source_df[relevant_cols]
    except KeyError: return pd.DataFrame([{t13_names['col1']: 'Base', t13_names['col2']: '0'}])
    base = t13['dq1b'].isin([1, 2]).sum()
    if base == 0: return pd.DataFrame([{t13_names['col1']: 'Base', t13_names['col2']: '0'}])
    dq2b_codes = t13['dq2b'].dropna().unique(); dq2a_codes = t13.filter(regex=r'^dq2a_\d+').columns.str.replace('dq2a_', '').astype(float).astype(int); all_bike_codes = np.union1d(dq2b_codes, dq2a_codes)
    results = []
    for code in all_bike_codes:
# --- START: New logic for unique user counting ---

        # Create a boolean mask for users who REPLACED this bike
        mask_replacement = (t13['dq2b'] == code)

        # Create a boolean mask for users who ALREADY OWNED this bike during an addition
        # Default to all False in case the column doesn't exist
        mask_addition = pd.Series([False] * len(t13), index=t13.index)
        dq2a_col_name = f'dq2a_{int(code)}'
        if dq2a_col_name in t13.columns:
            mask_addition = (t13[dq2a_col_name] == 1)

        # Combine the masks with a logical OR (|) to find users who are True in EITHER mask.
        # .sum() then counts the number of unique users who satisfy the condition.
        total_numerator = (mask_replacement | mask_addition).sum()
        percentage = (total_numerator / base) * 100
        
        # --- END: New logic ---
        results.append({'Code': code, t13_names['col2']: percentage})
    if not results: return pd.DataFrame([{t13_names['col1']: 'Base', t13_names['col2']: str(base)}])
    final_table = pd.DataFrame(results); bike_name_mapper = dict(zip(map_df['codes'], map_df['bike_names'])); brand_mapper = dict(zip(map_df['codes'], map_df['brands']))
    final_table[t13_names['col1']] = final_table['Code'].map(bike_name_mapper); final_table['Brand'] = final_table['Code'].map(brand_mapper)
    final_table = final_table[final_table[t13_names['col2']] > 0].copy()
    if final_table.empty: return pd.DataFrame([{t13_names['col1']: 'Base', t13_names['col2']: str(base)}]).drop(columns=['Brand'], errors='ignore')
    brand_totals = final_table.groupby('Brand')[t13_names['col2']].sum().sort_values(ascending=False); brand_order = brand_totals.index.tolist()
    final_table['Brand'] = pd.Categorical(final_table['Brand'], categories=brand_order, ordered=True); final_table.sort_values(by=['Brand', t13_names['col2']], ascending=[True, False], inplace=True)
    final_table = pd.concat([pd.DataFrame([{t13_names['col1']: 'Base', 'Brand': '', t13_names['col2']: base}]), final_table], ignore_index=True)
    def format_values(value):
        if isinstance(value, (int, float)):
            if value == int(value): return f'{int(value)}'
            else: return f'{value:.2f}'
        return value
    final_table[t13_names['col2']] = final_table[t13_names['col2']].apply(format_values)
    return final_table.drop(columns=['Brand','Code'])

def t11_calculate_brand_acquisition_share(source_df, map_df, column_config):
    t11_names = column_config['t11']
    base = source_df['dq1b'].isin([1, 2]).sum()
    if base == 0: return pd.DataFrame([{t11_names['col1']: 'Base', t11_names['col2']: '0'}])
    brand_to_codes_map = map_df.groupby('brands')['codes'].apply(list).to_dict(); results = []
    for brand, codes_for_brand in brand_to_codes_map.items():
        mask_replacement = source_df['dq2b'].isin(codes_for_brand); relevant_addition_cols = [f'dq2a_{int(code)}' for code in codes_for_brand if f'dq2a_{int(code)}' in source_df.columns]
        if relevant_addition_cols:
            mask_addition = (source_df[relevant_addition_cols] == 1).any(axis=1); combined_mask = mask_replacement | mask_addition
        else: combined_mask = mask_replacement
        net_buyers = combined_mask.sum(); percentage = (net_buyers / base) * 100
        results.append({t11_names['col1']: brand, t11_names['col2']: percentage})
    final_table = pd.DataFrame(results); final_table = final_table[final_table[t11_names['col2']] > 0].copy(); final_table.sort_values(by=t11_names['col2'], ascending=False, inplace=True)
    final_table = pd.concat([pd.DataFrame([{t11_names['col1']: 'Base', t11_names['col2']: base}]), final_table], ignore_index=True)
    def format_values(value):
        if isinstance(value, (int, float)):
            if value == int(value): return f'{int(value)}'
            else: return f'{value:.2f}'
        return value
    final_table[t11_names['col2']] = final_table[t11_names['col2']].apply(format_values)
    return final_table