import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
import os

# Configure page
st.set_page_config(
    page_title="Predictive Maintenance for Solar Power Plants",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1a5276;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .slide-header {
        font-size: 2.5rem;
        color: #1a5276;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 1.5rem;
    }
    .highlight {
        color: #3498db;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .problem-box {
        background-color: #fdf2f2;
        border-left: 4px solid #e74c3c;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .future-box {
        background-color: #f0f9f0;
        border-left: 4px solid #2ecc71;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📋 Navigation")
slides = [
    "Title Slide",
    "Introduction & Background", 
    "Problem Statement & Objectives",
    "Literature Review",
    "Methodology: CRISP-DM",
    "Dataset & Data Preparation",
    "Machine Learning Models",
    "Results & Evaluation",
    "Key Findings & Implications",
    "Limitations & Future Work",
    "Conclusion"
]

selected_slide = st.sidebar.selectbox("Select a slide:", slides)

# Helper function to load images
def load_image(image_path):
    try:
        return Image.open(image_path)
    except:
        st.error(f"Could not load image: {image_path}")
        return None

# Slide 1: Title Slide
if selected_slide == "Title Slide":
    st.markdown('<h1 class="main-header">Predictive Maintenance System for Solar Power Plant with Machine Learning</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        solar_plant_img = load_image("images/solar_plant.jpg")
        if solar_plant_img:
            st.image(solar_plant_img, caption="Solar Power Plant", use_container_width=True)
    
    st.markdown("### By: Muhammad Hakim bin Muhammad Taufik")
    st.markdown("### Universiti Teknologi Petronas")
    st.markdown("### May 2025")

# Slide 2: Introduction & Background
elif selected_slide == "Introduction & Background":
    st.markdown('<h2 class="slide-header">Introduction & Background</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### Global Shift Toward Renewable Energy
        - Increasing transition from fossil fuels to sustainable alternatives
        - Focus on reducing greenhouse gas emissions
        
        #### Solar Photovoltaic (PV) Power Plants
        - Crucial solution in renewable energy transition
        - Sustainable electricity generation from sunlight
        
        #### Maintenance Challenges
        - Inverter failures 
        - Panel degradation
        - Environmental fluctuations affecting performance
        
        #### Solar System Components
        - Solar panels (photovoltaic cells)
        - Inverters (DC to AC conversion)
        - Controllers and grid connections
        """)
        
        st.info("According to the International Energy Agency (IEA), achieving net zero emissions by 2050 requires a 70% contribution from wind and solar power.")
    
    with col2:
        solar_plant_img = load_image("C:/Users/User/My Files/CURRENT/FYP_1/streamlit_presentation/images/solar_plant.jpg")
        if solar_plant_img:
            st.image(solar_plant_img, caption="Solar Power Plant Aerial View", use_container_width=True)

# Slide 3: Problem Statement & Objectives
elif selected_slide == "Problem Statement & Objectives":
    st.markdown('<h2 class="slide-header">Problem Statement & Objectives</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Problem Statement section
        st.subheader("Problem Statement")
        st.markdown(
            """
            Traditional maintenance systems that rely on regular intervals or reactive fixes frequently result in:
            - Downtime and lower productivity  
            - Excessive maintenance expenses
            - Reduced energy yield and system longevity 
            - According to Hind(2024) **Potential yearly loss of $14.5 billion by 2024** in PV industry without optimization  
            - According to Benabbou et al. (2019), predictive maintenance has the potential to save maintenance costs by 10% to 40%. 
            """
        )

        # Research Objectives section
        st.subheader("Research Objectives")
        st.markdown(
            """
            1. **Primary Objective**: Develop and implement machine learning-based predictive maintenance that analyses historical data from solar power plant components to predict maintenance needs based on inverter efficiency degradation.  

            2. **Secondary Objective**: Compare and find the best model to predict maintenance based on inverter efficiency degradation.
            """
        )

    with col2:
        inverter_img = load_image("images/isometric-solar-panel-cell-system-260nw-1863823798.jpg")
        if inverter_img:
            st.image(
                inverter_img,
                caption="Solar Inverter - Critical Component for DC to AC Conversion",
                use_container_width=True
            )

# Slide 4: Literature Review
elif selected_slide == "Literature Review":
    st.markdown("## Literature Review")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        
        
        phases = [
            ("Solar PV Growth and Importance", "Solar PV is a rapidly expanding renewable energy source, crucial for energy transition and achieving net-zero emissions. Efficient operation is vital for optimizing energy yield and longevity."),
            ("Common PV System Failures", "V systems are susceptible to operational failures such as inverter overheating, string disconnections, partial shade, soiling, snow, and degradation mechanisms "),
            ("Anomaly Detection in Solar Plants", "Agussalim (2024) studied predictive maintenance focusing on anomaly detection using LSTM-AE models, which excelled in capturing temporal dependencies and complex patterns in PV datasets."),
            ("Solar Irradiance and Temperature Prediction", "Jose et al. (2022) reviewed models (ANN, SVM, ANFIS) for predicting solar irradiance and temperaturein order to capture stochastic behaviors and non-linearities in weather variables"),
            ("Predicting PV Underperformance", "Demetris et al. (2024) used XGBoost, one-class SVM, and Facebook Prophet (FBP) algorithm to predict PV underperformance, detect faults, and generate predictive maintenance alerts. Their routine could simulate PV system behavior under dynamic conditions and generate alerts up to 7 days in advance."),
     
        ]
        
        for i, (phase, description) in enumerate(phases, 1):
            st.subheader(f"{"-"} {phase}")
            st.write(description)
    
    with col2:
        crisp_dm_img = load_image("images/lt.jpg")
        if crisp_dm_img:
            st.image(
                crisp_dm_img,
                caption="CRISP-DM Methodology Framework",
                use_container_width=True
            )
        
        

# Slide 4: Methodology
elif selected_slide == "Methodology: CRISP-DM":
    st.markdown("## Methodology: CRISP-DM")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(
            "The study follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** methodology. The Deployment phase is not included in this project.:"
        )
        
        phases = [
            ("Business Understanding", "The primary goal is to predict the need for maintenance based on inverter efficiency degradation. The system aims to generate a binary maintenance flag (indicating whether an inverter is likely to require maintenance soon) from historical inverter and weather data."),
            ("Data Understanding", "The study assumes the dataset is a compilation of readings from 22 underperforming inverters within a solar power plant."),
            ("Data Preparation", "Filtering non-operational periods, feature engineering"),
            ("Modeling", "Training Random Forest, KNN, and SVM classifiers"),
            ("Evaluation", "Assessing model performance using metrics such as accuracy, precision, recall, and F1-score."),
            
        ]
        
        for i, (phase, description) in enumerate(phases, 1):
            st.subheader(f"{i}. {phase}")
            st.write(description)
    
    with col2:
        crisp_dm_img = load_image("images/crisp_dm.png")
        if crisp_dm_img:
            st.image(
                crisp_dm_img,
                caption="CRISP-DM Methodology Framework",
                use_container_width=True
            )
        
        st.info(
            "**Classification Strategy**: Defining inverter efficiency threshold (AC_POWER / DC_POWER) to detect early degradation signs")


# Slide 5: Dataset & Data Preparation
elif selected_slide == "Dataset & Data Preparation":
    st.markdown('<h2 class="slide-header">Dataset & Data Preparation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Dataset Information")
        st.markdown("""
        - **Source**: Solar plant in India
        - **Duration**: 34-day period
        - **Interval**: 15-minute intervals
        """)
        
        st.markdown("#### 📋 Data Types")
        data_types = pd.DataFrame({
            'Data Type': ['Plant Generation Data', 'Plant Weather Data'],
            'Key Features': ['AC_POWER, DC_POWER, DAILY_YIELD', 'AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION']
        })
        st.table(data_types)
        
        st.markdown("#### ⚙️ Data Preprocessing Steps")
        steps = [
            "Filtering Non-operational Periods",
            "Feature Engineering", 
            "Creating Binary Maintenance Labels",
            "Data Merging"
        ]
        
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}.** {step}")
    
    with col2:
        # Create dataset structure chart
        fig = go.Figure(data=[
            go.Bar(name='Number of Features', 
                   x=['Generation Data', 'Weather Data', 'Merged Dataset'], 
                   y=[3, 3, 6],
                   marker_color='#3498db')
        ])
        
        fig.update_layout(
            title='Dataset Structure',
            xaxis_title='Data Type',
            yaxis_title='Number of Features',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Slide 6: Machine Learning Models
elif selected_slide == "Machine Learning Models":
    st.markdown("## Machine Learning Models")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        models = [
            {
                "name": "🌳 Random Forest Classifier",
                "description": "An ensemble learning method that constructs multiple decision trees during training. Excellent for handling non-linear relationships.",
                "strength": "Robust against overfitting and handles high-dimensional data well."
            },
            {
                "name": "🔗 K-Nearest Neighbors (KNN)",
                "description": "A non-parametric method that classifies based on the majority class of its k nearest neighbors in the feature space.",
                "strength": "Simple implementation and adaptability to complex decision boundaries."
            },
            {
                "name": "📐 Support Vector Machine (SVM)",
                "description": "Finds the hyperplane that best separates classes in a high-dimensional space, maximizing the margin between classes.",
                "strength": "Effective in high-dimensional spaces and with clear margin of separation."
            }
        ]
        
        for model in models:
            st.subheader(model['name'])
            st.write(model['description'])
            st.markdown(f"**Key strength:** {model['strength']}")
    
    with col2:
        # Radar chart for model comparison
        categories = ['Accuracy', 'Training Speed', 'Prediction Speed', 'Interpretability', 'Handling Non-linearity']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[9.8, 7, 8, 6, 9],
            theta=categories,
            fill='toself',
            name='Random Forest',
            line_color='#2ecc71'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[8.5, 9, 6, 8, 7],
            theta=categories,
            fill='toself',
            name='KNN',
            line_color='#3498db'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[8.7, 6, 7, 5, 8],
            theta=categories,
            fill='toself',
            name='SVM',
            line_color='#9b59b6'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title="Model Characteristics Comparison",
            height=400
        )
        
        
        st.plotly_chart(fig, use_container_width=True)


# Slide 7: Results & Evaluation
elif selected_slide == "Results & Evaluation":
    st.markdown("## Results & Evaluation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Performance")

        st.markdown("**-🏆 Random Forest Classifier:** Highest accuracy at ~98%")
        st.write("Strong performance across precision, recall, and F1-score metrics")

        st.markdown("**-K-Nearest Neighbors (KNN):** Good performance but less accurate than Random Forest")

        st.markdown("**-Support Vector Machine (SVM):** Competitive performance but more computationally intensive")


    with col2:
        # Create performance comparison chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        random_forest = [0.98, 0.97, 0.98, 0.97]
        knn = [0.94, 0.93, 0.94, 0.93]
        svm = [0.92, 0.91, 0.92, 0.91]
        
        fig = go.Figure(data=[
            go.Bar(name='Random Forest', x=metrics, y=random_forest, marker_color='#3498db'),
            go.Bar(name='KNN', x=metrics, y=knn, marker_color='#2ecc71'),
            go.Bar(name='SVM', x=metrics, y=svm, marker_color='#9b59b6')
        ])
        
        fig.update_layout(
            title='Model Performance Metrics',
            xaxis_title='Metrics',
            yaxis_title='Score',
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 1])
        )
        crisp_dm_img = load_image("images/result.png")
        if crisp_dm_img:
            st.image(
                crisp_dm_img,
                caption="Result Comparison",
                use_container_width=True
            )
        st.plotly_chart(fig, use_container_width=True)


# Slide 8: Key Findings & Implications
elif selected_slide == "Key Findings & Implications":
    st.markdown("## Key Findings & Implications")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Key Research Findings")
        
        findings = [
            ("✅ High Model Accuracy", "Random Forest Classifier achieved ~98% accuracy in predicting maintenance needs based on inverter efficiency degradation."),
            ("📈 Operational Efficiency", "Predictive maintenance can increase operational efficiency by detecting early signs of inverter degradation before critical failure occurs."),
            ("⏰ Reduced Downtime", "Early detection allows for scheduled maintenance, reducing unexpected downtime by up to 10-40% compared to traditional maintenance approaches."),
            ("💰 Cost Savings", "Implementation of predictive maintenance can recover an average energy of 5.27% for a typical 16.1 MWp PV plant, equivalent to $10,000 per MW annually according to Hind (2024).")
        ]
        
        for icon_title, description in findings:
            st.markdown(f"**{icon_title}**")
            st.write(description)
    

        


# Slide 9: Limitations & Future Work
elif selected_slide == "Limitations & Future Work":
    st.markdown("## Limitations & Future Work")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚠️ Current Limitations")
        
        limitations = [
            "**Limited Dataset Duration**: The 34-day period may not capture all seasonal variations and proper health status of the inverters.",
            "**Target Variable**: The dataset lacks a specific failure event or labeled consequence that firmly defines the point of inverter breakdown or maintenance intervention.",
            "**Binary Classification**: May oversimplify the degradation process."
        ]
        
        for limitation in limitations:
            st.markdown(limitation)
    
    with col2:
        st.subheader("💡 Future Work Recommendations")
        
        future_work = [
            "**Real-time Monitoring**: Deploy models in real-time environments for immediate alerts.",
            "**Deep Learning Techniques**: Explore LSTM and RNN for improved temporal modeling.",
            "**Input Data**: To include inverter-specific logs, firmware upgrades, ambient temperature, module soiling, and maintenance history, all of which may affect inverter performance and failure timelines.",
            "**Days Until Failure**: To develop a regression-based model that forecasts the number of days until a failure occurs ."
        ]
        
        for work in future_work:
            st.markdown(work)
    
    # Limitations vs Future Potential Chart
    st.subheader("Limitations vs Future Potential")
    
    categories = ['Generalizability', 'Temporal Coverage', 'Feature Richness', 'Classification Granularity', 'Real-time Capability']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[3, 4, 5, 4, 2],
        theta=categories,
        fill='toself',
        name='Current Study Limitations',
        line_color='#e74c3c'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[8, 9, 8, 7, 9],
        theta=categories,
        fill='toself',
        name='Future Work Potential',
        line_color='#2ecc71'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=True,
        title="Limitations vs Future Potential",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)



# Slide 10: Conclusion
elif selected_slide == "Conclusion":
    st.markdown("## Conclusion")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        conclusions = [
            ("Research Contributions", 
             "Successfully developed a machine learning-based predictive maintenance system for solar power plants with a focus on inverter efficiency degradation detection."),
            ("Impact on Maintenance", 
             "The Random Forest model achieved ~98% accuracy, demonstrating that machine learning can effectively predict maintenance needs, reducing downtime and costs."),
            ("Broader Applications", 
             "The methodology can be extended to other renewable energy systems, contributing to the global transition toward sustainable energy sources.")
        ]

        for title, description in conclusions:
            st.subheader(title)
            st.write(description)
    
    with col2:
        solar_plant_img = load_image("images/premium.jpg")
        if solar_plant_img:
            st.image(solar_plant_img, caption="Solar Power Plant", use_container_width=True)
        
        st.success(
            '"Predictive maintenance through machine learning represents a significant step forward in optimizing renewable energy systems, ensuring their reliability and efficiency."')





# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About This Presentation")
st.sidebar.info("""
This interactive presentation is based on Muhammad Hakim's research on predictive maintenance for solar power plants using machine learning techniques.

**Key Highlights:**
- 98% accuracy with Random Forest
- CRISP-DM methodology
- Focus on inverter efficiency
- Real-world application potential
""")



