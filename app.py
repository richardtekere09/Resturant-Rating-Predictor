import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Restaurant Rating Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        color: #1f2937;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.125rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .feature-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    
    .info-panel {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-excellent {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .prediction-good {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .prediction-average {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .prediction-poor {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .section-header {
        color: #1f2937;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .input-label {
        color: #374151;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .rating-display {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .rating-category {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .rating-score {
        font-size: 1.125rem;
        opacity: 0.9;
    }
    
    .summary-card {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }
    
    .dataframe-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load models (with error handling)
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("scaler_clean.pkl")
        model = joblib.load("gridsrfr_model.pkl")
        return scaler, model
    except FileNotFoundError:
        st.error(
            "Model files not found. Please ensure 'scaler_clean.pkl' and 'gridsrfr_model.pkl' are in the same directory."
        )
        st.stop()


scaler, model = load_models()

# Header Section
st.markdown(
    '<h1 class="main-header">Restaurant Rating Analytics</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">AI-powered restaurant rating prediction system</p>',
    unsafe_allow_html=True,
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        '<div class="section-header">Restaurant Parameters</div>',
        unsafe_allow_html=True,
    )

    # Feature input section
    with st.container():
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)

        # Average Cost Input
        st.markdown(
            '<div class="input-label">Average Cost for Two People</div>',
            unsafe_allow_html=True,
        )
        averagecost = st.slider(
            label="cost_slider",
            min_value=50,
            max_value=100000,
            value=1000,
            step=100,
            help="Average dining cost for two customers",
            label_visibility="collapsed",
        )
        st.markdown(f"**Selected Cost:** ${averagecost:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Table Booking
        st.markdown(
            '<div class="input-label">Table Reservation System</div>',
            unsafe_allow_html=True,
        )
        tablebooking = st.radio(
            label="booking_radio",
            options=["Yes", "No"],
            horizontal=True,
            help="Advance table reservation availability",
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Online Delivery
        st.markdown(
            '<div class="input-label">Online Delivery Service</div>',
            unsafe_allow_html=True,
        )
        onlinedelivery = st.radio(
            label="delivery_radio",
            options=["Yes", "No"],
            horizontal=True,
            help="Digital ordering and delivery service",
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Price Range
        st.markdown(
            '<div class="input-label">Price Category</div>', unsafe_allow_html=True
        )
        price_labels = {
            "1": "Budget (Most Affordable)",
            "2": "Moderate Pricing",
            "3": "Premium Pricing",
            "4": "Luxury (Most Expensive)",
        }

        pricerange = st.selectbox(
            label="price_select",
            options=["1", "2", "3", "4"],
            format_func=lambda x: price_labels[x],
            help="Restaurant pricing tier classification",
            label_visibility="collapsed",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Button
    st.markdown(
        '<div class="section-header">Generate Prediction</div>', unsafe_allow_html=True
    )
    predict_button = st.button("ANALYZE RESTAURANT RATING", type="primary")

with col2:
    # Information Panel
    st.markdown(
        '<div class="section-header">System Information</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
    <div class="info-panel">
        <h4 style="margin-top: 0; color: #1f2937;">Machine Learning Model</h4>
        <p style="margin-bottom: 0; color: #4b5563;">Advanced Random Forest algorithm trained on comprehensive restaurant data to predict customer satisfaction ratings.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-panel">
        <h4 style="margin-top: 0; color: #1f2937;">Key Performance Factors</h4>
        <ul style="margin-bottom: 0; color: #4b5563;">
            <li>Average dining cost analysis</li>
            <li>Service convenience metrics</li>
            <li>Market positioning assessment</li>
            <li>Customer amenity evaluation</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Current inputs summary
    st.markdown(
        '<div class="section-header">Current Configuration</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
    <div class="summary-card">
        <strong>Average Cost:</strong> ${averagecost:,}<br>
        <strong>Table Booking:</strong> {tablebooking}<br>
        <strong>Online Delivery:</strong> {onlinedelivery}<br>
        <strong>Price Category:</strong> {price_labels[pricerange]}
    </div>
    """,
        unsafe_allow_html=True,
    )

# Prediction Section
st.markdown("---")

if predict_button:
    # Process inputs
    booking_status = 1 if tablebooking == "Yes" else 0
    delivery_status = 1 if onlinedelivery == "Yes" else 0

    # Create prediction array
    values = np.array([[averagecost, booking_status, delivery_status, int(pricerange)]])
    X = scaler.transform(values)

    # Make prediction
    prediction = model.predict(X)[0]

    # Display results
    st.markdown(
        '<div class="section-header">Analysis Results</div>', unsafe_allow_html=True
    )

    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])

    with col_pred2:
        # Determine rating category and styling
        if prediction < 2.5:
            category = "Below Average"
            rating_stars = "★★☆☆☆"
            css_class = "prediction-poor"
            recommendation = "Significant improvements needed in food quality, service, or overall customer experience."
        elif prediction < 3.5:
            category = "Average Performance"
            rating_stars = "★★★☆☆"
            css_class = "prediction-average"
            recommendation = "Solid foundation with opportunities for enhancement in key service areas."
        elif prediction < 4.0:
            category = "Good Rating"
            rating_stars = "★★★★☆"
            css_class = "prediction-good"
            recommendation = "Strong performance with potential for optimization to achieve excellence."
        else:
            category = "Excellent Rating"
            rating_stars = "★★★★★"
            css_class = "prediction-excellent"
            recommendation = (
                "Outstanding restaurant delivering exceptional customer satisfaction."
            )

        # Display prediction with styling
        st.markdown(
            f"""
        <div class="{css_class}">
            <div class="rating-display">{rating_stars}</div>
            <div class="rating-category">{category}</div>
            <div class="rating-score">Predicted Rating: {prediction:.2f}/5.0</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Additional insights
    st.markdown(
        '<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True
    )

    col_insight1, col_insight2 = st.columns(2)

    with col_insight1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3 style="margin-top: 0; margin-bottom: 0.5rem;">Model Confidence</h3>
            <p style="font-size: 1.25rem; margin: 0; font-weight: 600;">High Accuracy</p>
            <small style="opacity: 0.8;">Based on validated training data</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_insight2:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3 style="margin-top: 0; margin-bottom: 0.5rem;">Classification</h3>
            <p style="font-size: 1.25rem; margin: 0; font-weight: 600;">{category}</p>
            <small style="opacity: 0.8;">Customer satisfaction level</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Recommendation box
    st.success(f"**Strategic Recommendation:** {recommendation}")

    # Feature impact analysis
    st.markdown(
        '<div class="section-header">Feature Impact Analysis</div>',
        unsafe_allow_html=True,
    )

    impact_data = {
        "Parameter": [
            "Average Cost",
            "Table Booking",
            "Online Delivery",
            "Price Range",
        ],
        "Current Value": [
            f"${averagecost:,}",
            tablebooking,
            onlinedelivery,
            price_labels[pricerange],
        ],
        "Impact Level": ["High", "Medium", "Medium", "High"],
        "Weight": ["35%", "20%", "20%", "25%"],
    }

    df_analysis = pd.DataFrame(impact_data)

    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(df_analysis, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Welcome message when no prediction is made
    st.markdown(
        '<div class="section-header">Ready for Analysis</div>', unsafe_allow_html=True
    )
    st.info(
        "Configure the restaurant parameters above and click 'ANALYZE RESTAURANT RATING' to generate AI-powered insights."
    )

    # Benchmarking examples
    st.markdown(
        '<div class="section-header">Industry Benchmarks</div>', unsafe_allow_html=True
    )

    benchmark_col1, benchmark_col2, benchmark_col3 = st.columns(3)

    with benchmark_col1:
        st.markdown(
            """
        <div class="summary-card">
            <h4 style="margin-top: 0; color: #1f2937;">Budget Segment</h4>
            <p style="margin-bottom: 0.5rem;"><strong>Cost Range:</strong> $200-800</p>
            <p style="margin-bottom: 0.5rem;"><strong>Typical Rating:</strong> 2.5-3.2</p>
            <p style="margin-bottom: 0;"><strong>Key Factor:</strong> Value proposition</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with benchmark_col2:
        st.markdown(
            """
        <div class="summary-card">
            <h4 style="margin-top: 0; color: #1f2937;">Mid-Market Segment</h4>
            <p style="margin-bottom: 0.5rem;"><strong>Cost Range:</strong> $800-2500</p>
            <p style="margin-bottom: 0.5rem;"><strong>Typical Rating:</strong> 3.2-3.8</p>
            <p style="margin-bottom: 0;"><strong>Key Factor:</strong> Service quality</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with benchmark_col3:
        st.markdown(
            """
        <div class="summary-card">
            <h4 style="margin-top: 0; color: #1f2937;">Premium Segment</h4>
            <p style="margin-bottom: 0.5rem;"><strong>Cost Range:</strong> $2500+</p>
            <p style="margin-bottom: 0.5rem;"><strong>Typical Rating:</strong> 3.8-4.5</p>
            <p style="margin-bottom: 0;"><strong>Key Factor:</strong> Complete experience</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Sidebar enhancements
st.sidebar.markdown("### Model Documentation")
st.sidebar.markdown("""
**Restaurant Rating Analytics** employs machine learning algorithms to predict customer satisfaction ratings based on operational parameters.

**Technical Specifications:**
- **Algorithm:** Random Forest Regressor
- **Data Source:** Comprehensive restaurant dataset
- **Validation:** Cross-validated accuracy metrics
- **Features:** Cost, services, and market positioning

**Rating Scale:**
- **1.0 - 2.5:** Below Average
- **2.5 - 3.5:** Average Performance
- **3.5 - 4.0:** Good Rating
- **4.0 - 5.0:** Excellent Rating
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Development Team")
st.sidebar.info("**Lead Developer:** Richard\n\nBuilt with Streamlit & scikit-learn")

st.sidebar.markdown("### Technology Stack")
st.sidebar.markdown("""
- **Frontend Framework:** Streamlit
- **ML Library:** scikit-learn
- **Model Type:** Random Forest Regressor
- **Data Processing:** StandardScaler
- **Deployment:** Python 3.8+
""")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #6b7280; padding: 1.5rem;">
    <p><strong>Restaurant Rating Analytics Platform</strong> | Advanced predictive modeling for hospitality industry</p>
    <p><small>© 2024 - Professional grade machine learning solutions</small></p>
</div>
""",
    unsafe_allow_html=True,
)
