import streamlit as st
import pandas as pd
import re
import json
import yaml
from PIL import Image # --- ADDED IMPORT

import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

st.set_page_config(layout="wide")

# Import all the processing functions from your functions.py file
from functions import (
    process_demographics,
    generate_dealership_summary_tables,
    t1_create_brand_summary_from_codes,
    create_bike_summary_t2,
    t3_calculate_brand_frequency,
    t4_calculate_bike_frequency,
    t5_calculate_distribution,
    t6_create_bike_summary_table,
    t9_create_usage_distribution_table,
    t10_t12_calculate_purchase_type_distribution,
    t13_calculate_addition_replacement,
    t11_calculate_brand_acquisition_share
)

# --- START: REVISED LOGIN FUNCTION ---
def login_page():
    """Renders the login page and handles authentication."""
    st.title("Dashboard Login")

    # --- Placeholders for logos ---
    # Make sure you have an 'assets' folder with 'logo1.png' and 'logo2.png'
    try:
        logo1 = Image.open("assets/logo1.png")
        logo2 = Image.open("assets/logo2.png")
    except FileNotFoundError:
        st.error("Logo files not found. Please ensure 'assets/logo1.png' and 'assets/logo2.png' exist.")
        logo1, logo2 = None, None

    with st.form("login_form"):
        username = st.text_input("Username").lower()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            # --- Credentials Check ---
            correct_username = "misdashboard@infoleap"
            correct_password = "MIS@INFOLEAP"
            if username == correct_username and password == correct_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect username or password")

    if logo1 and logo2:
        col1, col2 = st.columns(2)
        with col1:
            # CHANGE 1: Switched to use_container_width and then decided to use fixed width for consistency.
            # CHANGE 2: Set a fixed width to make logos appear more uniform.
            st.image(logo1, width=250)
        with col2:
            st.image(logo2, width=250)

# --- END: REVISED LOGIN FUNCTION ---

# --- START: WRAPPED DASHBOARD CODE ---
def main_dashboard():
    """The main dashboard application, shown after successful login."""
    # --- DATA & CONFIG LOADING ---
    @st.cache_data
    def load_data():
        """Loads all data and configuration files from Google Drive."""
        try:
            # Load credentials from Streamlit secrets
            creds_json = {
                "type": st.secrets["gcp_service_account"]["type"],
                "project_id": st.secrets["gcp_service_account"]["project_id"],
                "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
                "private_key": st.secrets["gcp_service_account"]["private_key"].replace('\\n', '\n'),
                "client_email": st.secrets["gcp_service_account"]["client_email"],
                "client_id": st.secrets["gcp_service_account"]["client_id"],
                "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
                "token_uri": st.secrets["gcp_service_account"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"]
            }
    
            creds = service_account.Credentials.from_service_account_info(creds_json)
            service = build('drive', 'v3', credentials=creds)
    
            # File IDs have been populated from your links
            file_ids = {
                'data': '1uD7X-mOpRQXV4-uiKSmCuzjIAVs9OvoJ',
                'mapping': '1XU9rO3YV9lr-yDT7o7ag-DxOXcIm6fqQ',
                'titles': '11h9UYeRer2tTaBgrtX_Szcbc17R2kIc_',
                'filters': '197SMbuazbTJjWUGFNPoy3NhV6Os7jaQh',
                'columns': '1xUWY9ft-64FgwPprrRJAll1XG2GAUgGB'
            }
    
            def download_file(file_id):
                request = service.files().get_media(fileId=file_id)
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                file_stream.seek(0)
                return file_stream
    
            # Download and load all files
            data = pd.read_excel(download_file(file_ids['data']))
    
            map_sheets = {"aq3_po": "aq3_po", "aq5a": "aq5a", "aq5b_po": "aq5b_po", "aq6": "aq6", "q5": "q5", "q6a": "q6a", "dq1a": "dq1a", "dq1b": "dq1b", "dq2b": "dq2b", "dq2a": "dq2a", "dq3": "dq3", "dq4": "dq4", "dq6": "dq6", "dq7": "dq7"}
            with pd.ExcelFile(download_file(file_ids['mapping'])) as xls:
                maps = {name: pd.read_excel(xls, sheet_name=sheet) for name, sheet in map_sheets.items()}
    
            titles = json.load(download_file(file_ids['titles']))
            filter_config = yaml.safe_load(download_file(file_ids['filters']))
            columns = json.load(download_file(file_ids['columns']))
    
            return data, maps, filter_config, titles, columns
        except Exception as e:
            st.error(f"An error occurred while loading data from Google Drive: {e}")
            st.stop()
    def parse_codes(code_str):
        """Parses a string of codes into a list of integers."""
        if code_str is None:
            return []
        codes = []
        parts = str(code_str).split(',')
        for part in parts:
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    codes.extend(range(start, end + 1))
                except ValueError:
                    continue
            else:
                try:
                    codes.append(int(part))
                except ValueError:
                    continue
        return codes

    # Load all configurations and data
    data, maps, filter_config, TABLE_TITLES, COLUMN_NAMES = load_data()

    ZONE_MAP = filter_config.get('zone_map', {})
    SEGMENT_MAP = filter_config.get('segment_map', {})
    
    bike_filters_df = pd.DataFrame(filter_config.get('bike_filters', []))
    ALL_BRANDS = sorted(bike_filters_df['bike_name'].unique().tolist()) if not bike_filters_df.empty else []

    # --- UI SETUP ---
    st.title(TABLE_TITLES.get("main_title", "📊 Bike Market Analysis Dashboard"))

    # --- SIDEBAR FILTERS ---
    st.sidebar.title("Filters")

    # --- ADDED LOGOUT BUTTON ---
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    zone_selection = st.sidebar.selectbox("Select Zone", ["Overall"] + list(ZONE_MAP.keys()))
    segment_selection = st.sidebar.selectbox("Select User Segment", ["All Segments"] + list(SEGMENT_MAP.keys()))
    filter_path = st.sidebar.selectbox("Filter further by...", ["Overall", "CC & then Brand", "Specific Brand Directly"])

    # --- DYNAMIC FILTERS & DATA FILTERING LOGIC ---
    final_filtered_data = data.copy()

    if zone_selection != "Overall":
        zone_codes = parse_codes(ZONE_MAP.get(zone_selection))
        if zone_codes:
            final_filtered_data = final_filtered_data[final_filtered_data['s0'].isin(zone_codes)]

    seg_codes_to_filter_str = None
    if filter_path == "CC & then Brand":
        cc_choices = sorted(bike_filters_df['cc_category'].unique().tolist()) if not bike_filters_df.empty else []
        cc_choice = st.sidebar.selectbox("Select CC", cc_choices)
        
        brand_list_for_cc = sorted(bike_filters_df[bike_filters_df['cc_category'] == cc_choice]['bike_name'].tolist()) if not bike_filters_df.empty else []
        brand_list = [f"All {cc_choice} Brands"] + brand_list_for_cc
        brand_choice = st.sidebar.selectbox("Select Brand", brand_list)
        
        lookup_key = brand_choice if "All" not in brand_choice else cc_choice
        if segment_selection == "All Segments":
            seg_codes_to_filter_str = [seg_map.get(lookup_key) for seg_map in SEGMENT_MAP.values()]
        else:
            seg_codes_to_filter_str = SEGMENT_MAP.get(segment_selection, {}).get(lookup_key)

    elif filter_path == "Specific Brand Directly":
        brand_choice = st.sidebar.selectbox("Select Brand", ALL_BRANDS)
        if segment_selection == "All Segments":
             seg_codes_to_filter_str = [seg_map.get(brand_choice) for seg_map in SEGMENT_MAP.values()]
        else:
            seg_codes_to_filter_str = SEGMENT_MAP.get(segment_selection, {}).get(brand_choice)

    elif filter_path == "Overall" and segment_selection != "All Segments":
        seg_codes_to_filter_str = SEGMENT_MAP.get(segment_selection, {}).get('All')


    if seg_codes_to_filter_str:
        flat_codes = []
        items_to_process = seg_codes_to_filter_str if isinstance(seg_codes_to_filter_str, list) else [seg_codes_to_filter_str]
        
        for item in items_to_process:
            flat_codes.extend(parse_codes(item))
                
        unique_codes = set(flat_codes)
        if unique_codes:
            final_filtered_data = final_filtered_data[final_filtered_data['seg'].isin(unique_codes)]

    st.success("Dashboard updated for your selected filters ✅")

    # --- DATA PROCESSING ---
    fixed_columns = ['aq3_po', 'aq5b_po']
    pattern = r'(^aq5a_\d)|(^aq6_\d)'
    pattern_columns = [col for col in final_filtered_data.columns if re.search(pattern, col)]
    brands = final_filtered_data[fixed_columns + pattern_columns]

    # Call all functions
    t1 = t1_create_brand_summary_from_codes(brands, maps["aq3_po"], COLUMN_NAMES)
    t2 = create_bike_summary_t2(brands, maps["aq3_po"], COLUMN_NAMES)
    t3 = t3_calculate_brand_frequency(brands, maps["aq5a"], COLUMN_NAMES)
    t4 = t4_calculate_bike_frequency(brands, maps["aq5a"], COLUMN_NAMES)
    t5 = t5_calculate_distribution(brands, 'aq5b_po', maps["aq5b_po"], COLUMN_NAMES)
    t6 = t6_create_bike_summary_table(brands, maps["aq6"], COLUMN_NAMES)
    t7, t8 = generate_dealership_summary_tables(final_filtered_data, maps["q5"], maps["q6a"], COLUMN_NAMES)
    t9 = t9_create_usage_distribution_table(final_filtered_data, 'dq1a', maps["dq1a"], COLUMN_NAMES)
    t10 = t10_t12_calculate_purchase_type_distribution(final_filtered_data, maps["dq1b"], COLUMN_NAMES)
    t11 = t11_calculate_brand_acquisition_share(final_filtered_data, maps["dq2b"], COLUMN_NAMES)
    t13 = t13_calculate_addition_replacement(final_filtered_data, maps["dq2b"], COLUMN_NAMES)
    t14, t15, t16, t17 = process_demographics(final_filtered_data, maps["dq3"], maps["dq4"], maps["dq6"], maps["dq7"], COLUMN_NAMES)

    # --- DISPLAY TABS & TABLES ---
    brands_tab, dealership_tab, ar_tab, demo_tab = st.tabs(["Brands", "Dealership Visit", "Additional Replacement", "Demographic"])

    with brands_tab:
        b_own, b_con, m_con, t_drive = st.tabs(["Brands Owned", "Brand Considered", "Most Considered", "Test Drive"])
        with b_own:
            st.subheader(TABLE_TITLES.get('t1')); st.dataframe(t1, use_container_width=True)
            st.subheader(TABLE_TITLES.get('t2')); st.dataframe(t2, use_container_width=True)
        with b_con:
            st.subheader(TABLE_TITLES.get('t3')); st.dataframe(t3, use_container_width=True)
            st.subheader(TABLE_TITLES.get('t4')); st.dataframe(t4, use_container_width=True)
        with m_con:
            st.subheader(TABLE_TITLES.get('t5')); st.dataframe(t5, use_container_width=True)
        with t_drive:
            st.subheader(TABLE_TITLES.get('t6')); st.dataframe(t6, use_container_width=True)

    with dealership_tab:
        st.subheader(TABLE_TITLES.get('t7')); st.dataframe(t7, use_container_width=True)
        st.subheader(TABLE_TITLES.get('t8')); st.dataframe(t8, use_container_width=True)

    with ar_tab:
        u_ship, ar_dist, b_ar = st.tabs(["User Ship", "Additional Replacement Distribution", "Brands Of Additional+ Replacment"])
        with u_ship:
            st.subheader(TABLE_TITLES.get('t9')); st.dataframe(t9, use_container_width=True)
        with ar_dist:
            st.subheader(TABLE_TITLES.get('t10')); st.dataframe(t10, use_container_width=True)
        with b_ar:
            st.subheader(TABLE_TITLES.get('t11')); st.dataframe(t11, use_container_width=True)
            st.subheader(TABLE_TITLES.get('t13')); st.dataframe(t13, use_container_width=True)

    with demo_tab:
        edu, occ, inc, age = st.tabs(["Education", "Occupation", "Income", "Age"])
        with edu:
            st.subheader(TABLE_TITLES.get('t14')); st.dataframe(t14, use_container_width=True)
        with occ:
            st.subheader(TABLE_TITLES.get('t15')); st.dataframe(t15, use_container_width=True)
        with inc:
            st.subheader(TABLE_TITLES.get('t16')); st.dataframe(t16, use_container_width=True)
        with age:
            st.subheader(TABLE_TITLES.get('t17')); st.dataframe(t17, use_container_width=True)

# --- END: WRAPPED DASHBOARD CODE ---


# --- START: NEW MAIN APP CONTROLLER ---
# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Check authentication status and display the appropriate page
if not st.session_state.authenticated:
    login_page()
else:
    main_dashboard()
# --- END: NEW MAIN APP CONTROLLER ---