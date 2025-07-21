import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model using pickle
with open('best_xgb_regressor.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the normalization scaler
@st.cache_data
def load_scaler():
    """Load the MinMaxScaler used during training"""
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.error("scaler.pkl file not found. Please ensure the scaler file is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading scaler.pkl: {str(e)}")
        return None

# Initialize session state for storing predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Streamlit app title and configuration
st.set_page_config(page_title='Real Estate Price Prediction System', layout='wide')

# Custom CSS for dark mode and styling
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    
    /* Title container styling */
    .title-container {
        background-color: #90EE90;
        color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Predict button container */
    .predict-container {
        background-color: #90EE90;
        color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Column headers */
    .column-header {
        color: #90EE90;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }
    
    /* Input containers */
    .input-section {
        background-color: transparent;
        padding: 0px;
        border-radius: 0px;
        margin-bottom: 15px;
        border: none;
    }
    
    /* Override Streamlit's default styling */
    .stSelectbox label, .stSlider label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2d2d2d;
        color: white;
        border: 2px solid #90EE90;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #90EE90;
        color: #1e1e1e;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
        color: white;
    }
    
    /* Sidebar headers */
    .sidebar-header {
        color: #90EE90;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
        padding: 10px;
        background-color: #1e1e1e;
        border-radius: 5px;
    }
    
    /* Investment option containers */
    .investment-container {
        background-color: #1a1a1a;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #90EE90;
        border: 1px solid #404040;
    }
    
    /* Property feature list styling */
    .feature-list {
        background-color: #1a1a1a;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-family: monospace;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid #404040;
        white-space: pre-line;
    }
    
    /* Clear predictions button */
    .clear-button {
        background-color: #ff6b6b !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        width: 100% !important;
        margin-top: 10px !important;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background-color: #90EE90;
        color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown("""
<div class="title-container">
    <h1> Real Estate Price Prediction System</h1>
    <p>Get accurate property price predictions using advanced machine learning</p>
</div>
""", unsafe_allow_html=True)

# Load city-condition medians from pickle file
@st.cache_data
def load_city_condition_medians():
    """Load city-condition medians from pickle file"""
    try:
        loaded_df = pd.read_pickle('city_condition_medians.pkl')
        return loaded_df
    except FileNotFoundError:
        st.error("city_condition_medians.pkl file not found. Please ensure the file is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading city_condition_medians.pkl: {str(e)}")
        return None

# Function to calculate expected price per sqft
def calculate_expected_price_per_sqft_single(city, condition, medians_df):
  
    if medians_df is None:
        return 200  # Default value if data couldn't be loaded
    
    # Filter for the specific city and condition combination
    result = medians_df[(medians_df['city'] == city) & (medians_df['condition'] == condition)]
    
    if not result.empty:
        return result['expected_price_per_sqft'].iloc[0]
    else:
        return 200  # Default fallback

# Sidebar for investment comparison
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 Compare Investment Opportunities</div>', unsafe_allow_html=True)
    
    if len(st.session_state.predictions) == 0:
        st.info("Make some predictions first to compare investment opportunities!")
    else:
        st.write(f"**Total Predictions Made:** {len(st.session_state.predictions)}")
        
        # Investment preference selection
        st.markdown("### Select Investment Strategy")
        
        # Custom label with black text
        st.markdown('<p style="color: black; font-weight: bold; margin-bottom: 5px;">Choose your investment preference:</p>', unsafe_allow_html=True)
        
        investment_choice = st.selectbox(
            "Choose your investment preference:",
            ["Select an option", "Minimal Capital Investment", "High Capital Investment"],
            key="investment_selectbox",
            label_visibility="collapsed"
        )
        
        if investment_choice != "Select an option":
            # Find lowest and highest priced properties
            prices = [pred['price'] for pred in st.session_state.predictions]
            
            if investment_choice == "Minimal Capital Investment":
                min_price_idx = prices.index(min(prices))
                selected_property = st.session_state.predictions[min_price_idx]
                investment_type = "💰 Minimal Capital Investment"
                investment_description = "Lowest priced property from your predictions"
            else:  # High Capital Investment
                max_price_idx = prices.index(max(prices))
                selected_property = st.session_state.predictions[max_price_idx]
                investment_type = "🏆 High Capital Investment"
                investment_description = "Highest priced property from your predictions"
            
            # Display selected investment option
            st.markdown(f"""
            <div class="investment-container">
                <h4>{investment_type}</h4>
                <p style="color: #cccccc; margin-bottom: 10px;">{investment_description}</p>
                <h3 style="color: #90EE90;">${selected_property['price']:,.2f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display property features
            st.markdown("**Property Features:**")
            features_text = f"""bedrooms: {selected_property['features']['bedrooms']}
bathrooms: {selected_property['features']['bathrooms']}
floors: {selected_property['features']['floors']}
sqft_lot: {selected_property['features']['sqft_lot']:,}
sqft_living_above: {selected_property['features']['sqft_living_above']:,}
yr_built: {selected_property['features']['yr_built']}
yr_renovated: {selected_property['features']['yr_renovated']}
has_basement: {selected_property['features']['has_basement']}
view: {selected_property['features']['view']}
condition: {selected_property['features']['condition']}
city: {selected_property['features']['city']}
property_age: {selected_property['features']['property_age']} years
renewed_age: {selected_property['features']['renewed_age']} years
lot_to_living_ratio: {selected_property['features']['lot_to_living_ratio']:.2f}
expected_price_per_sqft: ${selected_property['features']['expected_price_per_sqft']:.2f}"""
            
            st.markdown(f'<div class="feature-list">{features_text}</div>', unsafe_allow_html=True)
        
        # Clear predictions button
        st.markdown("---")
        if st.button("🗑️ Clear All Predictions", key="clear_predictions"):
            st.session_state.predictions = []
            st.rerun()

# Load city-condition medians and scaler at the start of the app
city_condition_medians_df = load_city_condition_medians()
scaler = load_scaler()

# Display warning if scaler couldn't be loaded
if scaler is None:
    st.warning("⚠️ Normalization scaler could not be loaded. Predictions may be inaccurate.")

# City mapping
city_mapping = {
    "Shoreline": 1, "Kent": 2, "Bellevue": 3, "Redmond": 4, "Seattle": 5,
    "Maple Valley": 6, "North Bend": 7, "Lake Forest Park": 8, "Sammamish": 9,
    "Auburn": 10, "Des Moines": 11, "Bothell": 12, "Federal Way": 13,
    "Kirkland": 14, "Issaquah": 15, "Woodinville": 16, "Normandy Park": 17,
    "Fall City": 18, "Renton": 19, "Carnation": 20, "Snoqualmie": 21,
    "Duvall": 22, "Burien": 23, "Covington": 24, "Inglewood-Finn Hill": 25,
    "Kenmore": 26, "Newcastle": 27, "Black Diamond": 28, "Ravensdale": 29,
    "Clyde Hill": 30, "Algona": 31, "Mercer Island": 32, "Skykomish": 33,
    "Tukwila": 34, "Vashon": 35, "SeaTac": 36, "Enumclaw": 37,
    "Snoqualmie Pass": 38, "Pacific": 39, "Beaux Arts Village": 40,
    "Preston": 41, "Milton": 42, "Yarrow Point": 43, "Medina": 44
}

# Create two columns for the layout
col1, col2 = st.columns(2)

# Left column - Sliders
with col1:
    st.markdown('<div class="column-header"> Property Measurements & Details</div>', unsafe_allow_html=True)
    
    sqft_lot = st.slider('Square Feet of Lot', 0, 100000, 10000)
    sqft_living_above = st.slider('Square Feet of Living Above Ground', 0, 10000, 2000)
    yr_built = st.slider('Year Built', 1900, 2014, 2000)
    yr_renovated = st.slider('Year Renovated (0 if never renovated)', 0, 2014, 0)
    city = st.selectbox('Select City', list(city_mapping.keys()))
    has_basement = st.slider('Has Basement (0=No, 1=Yes)', 0, 1, 0)

# Right column - Select boxes
with col2:
    st.markdown('<div class="column-header"> Property Features</div>', unsafe_allow_html=True)
    
    bedrooms = st.selectbox('Number of Bedrooms', list(range(0, 11)), index=3)
    bathrooms_options = ["0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"]
    bathrooms = st.selectbox('Number of Bathrooms', bathrooms_options, index=3)
    floors = st.selectbox('Number of Floors', list(range(1, 5)), index=0)
    view = st.selectbox('View Rating (0-4)', list(range(5)), index=0)
    condition = st.selectbox('Property Condition (1-5)', list(range(1, 6)), index=2)

# Predict button with custom styling
st.markdown('<div class="predict-container">', unsafe_allow_html=True)
predict_clicked = st.button('Predict Property Price')
st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic and display
if predict_clicked:
    # Convert bathrooms from string to float
    bathrooms_float = float(bathrooms)
    
    # Extract city number from mapping
    city_number = city_mapping[city]

    # Calculate property and renewed age
    current_year = 2014  
    property_age = current_year - yr_built
    effective_yr_renovated = yr_built if yr_renovated == 0 else yr_renovated
    renewed_age = current_year - effective_yr_renovated
    
    # Calculate lot to living ratio
    if sqft_living_above > 0:
        lot_to_living_ratio = sqft_lot / sqft_living_above
    else:
        lot_to_living_ratio = 0
    
    # Calculate expected price per sqft
    expected_price_per_sqft = calculate_expected_price_per_sqft_single(city_number, condition, city_condition_medians_df)
    
    # DataFrame with preprocessed features
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms_float],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'view': [view],
        'condition': [condition],
        'city': [city_number],
        'sqft_living_above': [sqft_living_above],
        'expected_price_per_sqft': [expected_price_per_sqft],
        'property_age': [property_age],
        'renewed_age': [renewed_age],
        'lot_to_living_ratio': [lot_to_living_ratio], 
        'has_basement': [has_basement],    
    })
    
    # Make prediction
    try:
        # Apply normalization using the loaded scaler
        if scaler is not None:
            # Normalize the input data using the same scaler from training
            input_data_normalized = scaler.transform(input_data)
            prediction = model.predict(input_data_normalized)
        else:
            # If scaler couldn't be loaded, make prediction without normalization
            st.warning("Making prediction without normalization - results may be inaccurate!")
            prediction = model.predict(input_data)
        
        predicted_price = prediction[0]
        
        # Store prediction in session state
        prediction_data = {
            'price': predicted_price,
            'features': {
                'bedrooms': bedrooms,
                'bathrooms': bathrooms_float,
                'floors': floors,
                'sqft_lot': sqft_lot,
                'sqft_living_above': sqft_living_above,
                'yr_built': yr_built,
                'yr_renovated': yr_renovated,
                'has_basement': 'Yes' if has_basement else 'No',
                'view': view,
                'condition': condition,
                'city': city,
                'property_age': property_age,
                'renewed_age': renewed_age,
                'lot_to_living_ratio': lot_to_living_ratio,
                'expected_price_per_sqft': expected_price_per_sqft
            }
        }
        
        st.session_state.predictions.append(prediction_data)
        
        predicted_price_formatted = f"${predicted_price:,.2f}"
        
        # Display the prediction with custom styling
        st.markdown(f"""
        <div class="prediction-result">
            <h2> Predicted Property Price</h2>
            <h1>{predicted_price_formatted}</h1>
            <p>Based on the selected property features and location</p>
            <small>Prediction #{len(st.session_state.predictions)} saved for comparison</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Display calculated features for debugging/transparency
        with st.expander("Calculated Features"):
            st.write(f"**Property Age:** {property_age} years")
            st.write(f"**Renewed Age:** {renewed_age} years")
            st.write(f"**Lot to Living Ratio:** {lot_to_living_ratio:.2f}")
            st.write(f"**Expected Price per Sqft:** ${expected_price_per_sqft:.2f}")
            st.write(f"**Normalization Applied:** {'Yes' if scaler is not None else 'No'}")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Please check that your model expects the same features as provided.")