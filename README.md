# Solar Power Plant Predictive Maintenance - Streamlit Presentation

This is an interactive Streamlit application based on Muhammad Hakim's research on "Predictive Maintenance System for Solar Power Plant with Machine Learning."

## Features

- **Interactive Navigation**: Navigate through 10 presentation slides using the sidebar dropdown
- **Dynamic Visualizations**: Interactive charts and graphs using Plotly
- **Responsive Design**: Professional styling with custom CSS
- **Comprehensive Content**: All research findings, methodology, and results
- **Presentation Script**: Access to full speaking notes for each slide

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project files**
   ```bash
   # If you have the files, navigate to the project directory
   cd streamlit_presentation
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:
   ```bash
   pip install streamlit plotly pandas pillow
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - The application will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

## Application Structure

```
streamlit_presentation/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── images/               # Image assets
    ├── solar_plant.jpg
    ├── solar_inverter.png
    ├── crisp_dm.png
    ├── ml_comparison.png
    ├── predictive_maintenance.jpg
    └── solar_degradation.png
```

## Slide Contents

1. **Title Slide** - Research title and author information
2. **Introduction & Background** - Context on renewable energy and solar power
3. **Problem Statement & Objectives** - Research goals and challenges addressed
4. **Methodology: CRISP-DM** - Data mining methodology framework
5. **Dataset & Data Preparation** - Data sources and preprocessing steps
6. **Machine Learning Models** - Comparison of Random Forest, KNN, and SVM
7. **Results & Evaluation** - Model performance metrics and validation
8. **Key Findings & Implications** - Research outcomes and industry impact
9. **Limitations & Future Work** - Current constraints and future directions
10. **Conclusion** - Summary of contributions and final thoughts

## Key Features

### Interactive Charts
- **Radar Charts**: Model comparison and limitations analysis
- **Bar Charts**: Performance metrics and dataset structure
- **Dynamic Visualizations**: Responsive charts that adapt to screen size

### Navigation
- **Sidebar Navigation**: Easy slide selection
- **Presentation Script**: Access to detailed speaking notes
- **Professional Styling**: Custom CSS for academic presentation appearance

### Content Highlights
- **98% Accuracy**: Random Forest model performance
- **CRISP-DM Methodology**: Structured data mining approach
- **Real-world Application**: Practical implications for solar industry
- **Cost Savings**: $10,000 per MW annually potential savings

## Usage Tips

1. **Navigation**: Use the sidebar dropdown to move between slides
2. **Interactivity**: Hover over charts for detailed information
3. **Full Screen**: Use browser full-screen mode for presentation
4. **Script Access**: Click "View Full Presentation Script" for speaking notes
5. **Responsive**: Works on desktop, tablet, and mobile devices

## Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **Pandas**: Data manipulation and analysis
- **Pillow**: Image processing library

### Performance
- Optimized for fast loading and smooth navigation
- Responsive design for various screen sizes
- Efficient image loading and caching

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Missing Images**
   - Ensure all images are in the `images/` directory
   - Check file paths in the application

3. **Package Installation Issues**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

### Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari
- Edge

## Customization

### Styling
- Modify CSS in the `st.markdown()` sections of `app.py`
- Colors, fonts, and layout can be customized

### Content
- Update slide content by modifying the respective sections in `app.py`
- Add new slides by extending the navigation and content sections

### Charts
- Modify Plotly chart configurations for different visualizations
- Add new chart types as needed

## Support

For technical issues or questions about the research content, please refer to:
- Original dissertation document
- Presentation script for detailed explanations
- Research methodology and findings sections

## License

This application is created for educational and research purposes based on the academic work of Muhammad Hakim bin Muhammad Taufik at Universiti Teknologi Petronas.

