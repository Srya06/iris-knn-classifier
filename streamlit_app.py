# streamlit_app.py - Fixed Version
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Initialize session state
if 'classic_results' not in st.session_state:
    st.session_state.classic_results = None
if 'fast_results' not in st.session_state:
    st.session_state.fast_results = None

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier 🌸",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        margin-top: 1.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #4B8BBE;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #306998;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌸 Iris Flower Classification using KNN</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Classic vs Fast KNN Implementation | Mini Project
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", 
             caption="Iris Flowers", use_container_width=True)
    
    st.markdown("### 🎯 Project Settings")
    k_value = st.slider("Select K value:", 1, 15, 3, 
                       help="Number of nearest neighbors to consider")
    
    algorithm_choice = st.selectbox(
        "Choose KNN Algorithm:",
        ["Classic (Educational)", "Fast (scikit-learn)", "Compare Both"]
    )
    
    test_size = st.slider("Test Set Size (%):", 10, 40, 20)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.info("""
    **Iris Dataset:**
    - 150 samples total
    - 4 features per sample
    - 3 flower species
    - Balanced: 50 samples per class
    """)
    
    st.markdown("---")
    st.markdown("### 🌸 Flower Species")
    st.success("""
    1. **Setosa** - Small petals
    2. **Versicolor** - Medium petals  
    3. **Virginica** - Large petals
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Train Both")
    if st.button("Train Both KNN Models", use_container_width=True, key="train_both"):
        st.session_state.training_both = True

# Classic KNN Class
class ClassicKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = np.sqrt(np.sum((test_point - train_point) ** 2))
                distances.append((dist, self.y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            k_labels = [label for _, label in k_nearest]
            most_common = np.bincount(k_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)

# Load and prepare data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return iris, df

iris, df = load_data()

# Prepare data for modeling
X = df[iris.feature_names].values
y = df['species'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check if "Train Both" was clicked
if 'training_both' in st.session_state and st.session_state.training_both:
    # Train Classic KNN
    with st.spinner("Training Classic KNN..."):
        classic_knn = ClassicKNN(k=k_value)
        classic_knn.fit(X_train_scaled, y_train)
        start_time = time.time()
        y_pred_classic = classic_knn.predict(X_test_scaled)
        predict_time_classic = time.time() - start_time
        
        st.session_state.classic_results = {
            'accuracy': accuracy_score(y_test, y_pred_classic),
            'predict_time': predict_time_classic,
            'predictions': y_pred_classic,
            'model': classic_knn
        }
    
    # Train Fast KNN
    with st.spinner("Training Fast KNN..."):
        fast_knn = KNeighborsClassifier(n_neighbors=k_value, algorithm='kd_tree')
        fast_knn.fit(X_train_scaled, y_train)
        start_time = time.time()
        y_pred_fast = fast_knn.predict(X_test_scaled)
        predict_time_fast = time.time() - start_time
        
        st.session_state.fast_results = {
            'accuracy': accuracy_score(y_test, y_pred_fast),
            'train_time': 0.001,
            'predict_time': predict_time_fast,
            'predictions': y_pred_fast,
            'model': fast_knn,
            'probabilities': fast_knn.predict_proba(X_test_scaled)
        }
    
    st.success("✅ Both models trained successfully!")
    st.session_state.training_both = False
    st.rerun()

# Main content in tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset", "🔍 Classic KNN", "⚡ Fast KNN", 
    "📈 Comparison", "🌺 Classifier"
])

# Tab 1: Dataset Explorer
with tab1:
    st.markdown('<h2 class="sub-header">📊 Iris Dataset Explorer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.metric("Total Samples", df.shape[0])
        st.metric("Features", df.shape[1] - 2)
        st.metric("Classes", df['species'].nunique())
    
    with col2:
        st.markdown("### Class Distribution")
        fig = px.pie(df, names='species_name', 
                     title='Distribution of Iris Species',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📈 Feature Visualizations")
    
    # Feature distributions
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=iris.feature_names)
    
    for i, feature in enumerate(iris.feature_names):
        row = i // 2 + 1
        col = i % 2 + 1
        
        for species in df['species_name'].unique():
            species_data = df[df['species_name'] == species][feature]
            fig.add_trace(
                go.Box(y=species_data, name=species, showlegend=(i==0)),
                row=row, col=col
            )
    
    fig.update_layout(height=600, title_text="Feature Distributions by Species")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### 🔥 Feature Correlation")
    corr = df[iris.feature_names].corr()
    fig = px.imshow(corr, text_auto=True, 
                    color_continuous_scale='RdBu_r',
                    title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Classic KNN Implementation
with tab2:
    st.markdown('<h2 class="sub-header">🔍 Classic KNN Implementation (From Scratch)</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🚀 Train Classic KNN", key="train_classic", use_container_width=True):
            with st.spinner("Training Classic KNN..."):
                classic_knn = ClassicKNN(k=k_value)
                classic_knn.fit(X_train_scaled, y_train)
                
                # Time prediction
                start_time = time.time()
                y_pred_classic = classic_knn.predict(X_test_scaled)
                predict_time = time.time() - start_time
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred_classic)
                
                # Store in session state
                st.session_state.classic_results = {
                    'accuracy': accuracy,
                    'predict_time': predict_time,
                    'predictions': y_pred_classic,
                    'model': classic_knn
                }
                
                st.success(f"✅ Classic KNN trained! Accuracy: {accuracy:.2%}")
    
    if st.session_state.classic_results is not None:
        results = st.session_state.classic_results
        
        with col2:
            st.metric("Accuracy", f"{results['accuracy']:.2%}")
            st.metric("Prediction Time", f"{results['predict_time']:.4f}s")
        
        # Confusion Matrix
        st.markdown("### 📊 Performance Metrics")
        
        cm = confusion_matrix(y_test, results['predictions'])
        fig = px.imshow(cm, 
                      text_auto=True,
                      labels=dict(x="Predicted", y="Actual", color="Count"),
                      x=iris.target_names,
                      y=iris.target_names,
                      title="Confusion Matrix - Classic KNN",
                      color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show some predictions
        st.markdown("### 🔍 Sample Predictions")
        sample_df = pd.DataFrame({
            'Actual': [iris.target_names[i] for i in y_test[:10]],
            'Predicted': [iris.target_names[i] for i in results['predictions'][:10]],
            'Correct': ['✅' if a == p else '❌' for a, p in zip(y_test[:10], results['predictions'][:10])]
        })
        st.dataframe(sample_df, use_container_width=True)
        
        # How it works explanation
        with st.expander("ℹ️ How Classic KNN Works"):
            st.markdown("""
            **Classic KNN Algorithm:**
            1. **Lazy Learning**: Stores all training data
            2. **Distance Calculation**: For each test point, calculates distance to ALL training points
            3. **Find Neighbors**: Selects K nearest neighbors
            4. **Majority Vote**: Predicts the most common label among neighbors
            
            **Time Complexity**: O(n²) - Slow for large datasets
            **Best For**: Educational purposes and understanding KNN basics
            """)
    else:
        st.info("Click 'Train Classic KNN' button to train the model and see results.")

# Tab 3: Fast KNN Implementation
with tab3:
    st.markdown('<h2 class="sub-header">⚡ Fast KNN (scikit-learn Optimized)</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("⚡ Train Fast KNN", key="train_fast", use_container_width=True):
            with st.spinner("Training Fast KNN..."):
                fast_knn = KNeighborsClassifier(n_neighbors=k_value, algorithm='kd_tree')
                
                # Time training
                train_start = time.time()
                fast_knn.fit(X_train_scaled, y_train)
                train_time = time.time() - train_start
                
                # Time prediction
                start_time = time.time()
                y_pred_fast = fast_knn.predict(X_test_scaled)
                predict_time = time.time() - start_time
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred_fast)
                
                # Store in session state
                st.session_state.fast_results = {
                    'accuracy': accuracy,
                    'train_time': train_time,
                    'predict_time': predict_time,
                    'predictions': y_pred_fast,
                    'model': fast_knn,
                    'probabilities': fast_knn.predict_proba(X_test_scaled)
                }
                
                st.success(f"✅ Fast KNN trained! Accuracy: {accuracy:.2%}")
    
    if st.session_state.fast_results is not None:
        results = st.session_state.fast_results
        
        with col2:
            st.metric("Accuracy", f"{results['accuracy']:.2%}")
            st.metric("Prediction Time", f"{results['predict_time']:.6f}s")
        
        # Performance metrics
        st.markdown("### 📊 Detailed Performance")
        
        # Classification report
        report = classification_report(y_test, results['predictions'], 
                                     target_names=iris.target_names, 
                                     output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, results['predictions'])
        fig = px.imshow(cm, 
                      text_auto=True,
                      labels=dict(x="Predicted", y="Actual", color="Count"),
                      x=iris.target_names,
                      y=iris.target_names,
                      title="Confusion Matrix - Fast KNN",
                      color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance visualization
        st.markdown("### 📈 Decision Boundary Visualization")
        
        # Using first two features for 2D visualization
        fig = px.scatter(df, x='sepal length (cm)', y='sepal width (cm)',
                       color='species_name',
                       color_discrete_sequence=px.colors.qualitative.Set2,
                       title='Iris Data: Sepal Dimensions',
                       labels={'species_name': 'Species'})
        st.plotly_chart(fig, use_container_width=True)
        
        # How it works explanation
        with st.expander("ℹ️ How Fast KNN Works"):
            st.markdown("""
            **Optimized KNN Algorithm:**
            1. **KD-Tree**: Data organized in tree structure for faster search
            2. **Vectorized Operations**: Uses NumPy for efficient computations
            3. **Optimized Distance Calculations**: Precomputations and caching
            4. **C++ Backend**: Critical operations in C++ for speed
            
            **Time Complexity**: O(n log n) - Much faster for large datasets
            **Best For**: Production applications and real-time predictions
            """)
    else:
        st.info("Click 'Train Fast KNN' button to train the model and see results.")

# Tab 4: Comparison
with tab4:
    st.markdown('<h2 class="sub-header">📈 Classic vs Fast KNN Comparison</h2>', unsafe_allow_html=True)
    
    if st.session_state.classic_results is not None and st.session_state.fast_results is not None:
        classic = st.session_state.classic_results
        fast = st.session_state.fast_results
        
        # Comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy (Classic)", f"{classic['accuracy']:.2%}")
        with col2:
            st.metric("Accuracy (Fast)", f"{fast['accuracy']:.2%}")
        with col3:
            speed_ratio = classic['predict_time'] / fast['predict_time'] if fast['predict_time'] > 0 else 0
            st.metric("Speed Ratio", f"{speed_ratio:.1f}x")
        with col4:
            diff = abs(classic['accuracy'] - fast['accuracy'])
            st.metric("Accuracy Difference", f"{diff:.4f}")
        
        # Comparison chart
        st.markdown("### 📊 Performance Comparison")
        
        comparison_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Prediction Time (s)', 'Implementation'],
            'Classic KNN': [
                f"{classic['accuracy']:.2%}",
                f"{classic['predict_time']:.6f}",
                'From Scratch'
            ],
            'Fast KNN': [
                f"{fast['accuracy']:.2%}",
                f"{fast['predict_time']:.6f}",
                'scikit-learn'
            ]
        })
        
        st.dataframe(comparison_data, use_container_width=True)
        
        # Bar chart comparison
        fig = go.Figure(data=[
            go.Bar(name='Classic KNN', x=['Accuracy', 'Prediction Time'], 
                   y=[classic['accuracy'], classic['predict_time'] * 1000]),  # Convert to ms for visualization
            go.Bar(name='Fast KNN', x=['Accuracy', 'Prediction Time'], 
                   y=[fast['accuracy'], fast['predict_time'] * 1000])
        ])
        
        fig.update_layout(
            title='Side-by-Side Comparison',
            barmode='group',
            yaxis_title='Value',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 🏆 Recommendation")
        
        if classic['accuracy'] > fast['accuracy']:
            st.success("""
            **Classic KNN performs slightly better!**
            - Use for educational purposes
            - Good for understanding KNN fundamentals
            - Not recommended for large datasets
            """)
        else:
            st.success("""
            **Fast KNN is recommended!**
            - Better performance and speed
            - Production-ready implementation
            - Additional features (probabilities, different metrics)
            - Better scalability
            """)
        
        # Decision matrix
        st.markdown("### 🤔 When to Use Which?")
        
        decision_data = {
            'Factor': ['Learning/Education', 'Production Use', 'Large Datasets', 
                      'Need Probabilities', 'Customization Needs'],
            'Classic KNN': ['✅ Excellent', '❌ Not recommended', '❌ Poor', 
                          '❌ Manual implementation', '✅ Full control'],
            'Fast KNN': ['✅ Good', '✅ Excellent', '✅ Good', 
                        '✅ Built-in', '❌ Limited to library features']
        }
        
        st.table(pd.DataFrame(decision_data))
    
    else:
        st.warning("""
        **Run both Classic and Fast KNN first!**
        
        To see the comparison:
        1. Go to **Classic KNN** tab and click "Train Classic KNN"
        2. Go to **Fast KNN** tab and click "Train Fast KNN"
        3. Return here to see the comparison
        
        Or use the **"Train Both KNN Models"** button in the sidebar for quick setup.
        """)

# Tab 5: Interactive Classifier
with tab5:
    st.markdown('<h2 class="sub-header">🌺 Interactive Iris Classifier</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <b>Enter flower measurements below to classify the iris species!</b><br>
        All measurements should be in centimeters (cm).
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input sliders
        st.markdown("### 📏 Enter Measurements")
        
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1,
                               help="Length of sepal - usually 4.3 to 7.9 cm")
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1,
                              help="Width of sepal - usually 2.0 to 4.4 cm")
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8, 0.1,
                               help="Length of petal - usually 1.0 to 6.9 cm")
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1,
                              help="Width of petal - usually 0.1 to 2.5 cm")
    
    with col2:
        # Display flower image based on typical values
        st.markdown("### 🌼 Typical Ranges")
        
        # Determine which species is most likely based on petal measurements
        if petal_width < 0.6 and petal_length < 2:
            species_guess = "setosa"
            img_url = "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg"
            color = "#FF6B6B"
        elif petal_width < 1.8 and petal_length < 5:
            species_guess = "versicolor"
            img_url = "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"
            color = "#4ECDC4"
        else:
            species_guess = "virginica"
            img_url = "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
            color = "#45B7D1"
        
        st.markdown(f"""
        <div style="background-color: {color}20; padding: 15px; border-radius: 10px; border-left: 5px solid {color};">
            <h4 style="color: {color}; margin-top: 0;">Likely Species: {species_guess.title()}</h4>
            <p style="margin-bottom: 0;">Based on your inputs, this flower looks like <b>{species_guess.title()}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(img_url, caption=f"Iris {species_guess.title()}", use_container_width=True)
    
    # Classification button
    if st.button("🔍 Classify This Flower!", use_container_width=True, key="classify"):
        # Prepare input
        user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale input
        scaler = StandardScaler()
        scaler.fit(df[iris.feature_names].values)  # Fit on all data
        
        user_input_scaled = scaler.transform(user_input)
        
        # Make predictions
        predictions = {}
        probabilities = {}
        
        if st.session_state.classic_results is not None:
            classic_pred = st.session_state.classic_results['model'].predict(user_input_scaled)[0]
            predictions['Classic KNN'] = iris.target_names[classic_pred]
        
        if st.session_state.fast_results is not None:
            fast_pred = st.session_state.fast_results['model'].predict(user_input_scaled)[0]
            fast_proba = st.session_state.fast_results['model'].predict_proba(user_input_scaled)[0]
            predictions['Fast KNN'] = iris.target_names[fast_pred]
            probabilities = {iris.target_names[i]: f"{fast_proba[i]:.1%}" for i in range(3)}
        
        # Display results
        if predictions:
            st.markdown("### 🎯 Classification Results")
            
            result_cols = st.columns(len(predictions))
            
            for idx, (model_name, pred_species) in enumerate(predictions.items()):
                with result_cols[idx]:
                    colors = {
                        'setosa': '#FF6B6B',
                        'versicolor': '#4ECDC4', 
                        'virginica': '#45B7D1'
                    }
                    
                    st.markdown(f"""
                    <div style="background-color: {colors[pred_species]}20; 
                                padding: 20px; border-radius: 10px; 
                                border-left: 5px solid {colors[pred_species]};
                                text-align: center;">
                        <h3 style="color: {colors[pred_species]}; margin-top: 0;">
                            {pred_species.title()}
                        </h3>
                        <p><b>{model_name}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show probabilities if available
            if probabilities:
                st.markdown("### 📊 Confidence Scores (Fast KNN)")
                
                # Create gauge chart
                fig = go.Figure()
                
                for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
                    value = float(probabilities[species].strip('%')) / 100
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': species.title()},
                        domain={'row': 0, 'column': i},
                        gauge={'axis': {'range': [0, 1]},
                              'bar': {'color': colors[species]},
                              'steps': [
                                  {'range': [0, 0.33], 'color': "lightgray"},
                                  {'range': [0.33, 0.66], 'color': "gray"},
                                  {'range': [0.66, 1], 'color': "darkgray"}
                              ]}
                    ))
                
                fig.update_layout(
                    grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Flower information
            st.markdown("### 🌿 About the Predicted Species")
            
            species_info = {
                'setosa': {
                    'description': 'Iris setosa is a species in the genus Iris, native to northern regions.',
                    'characteristics': ['Smallest petals', 'Bright colors', 'Northern habitats'],
                    'habitat': 'Cool, northern regions'
                },
                'versicolor': {
                    'description': 'Iris versicolor is also known as the harlequin blueflag.',
                    'characteristics': ['Medium-sized petals', 'Varied colors', 'Wetland habitats'],
                    'habitat': 'Wetlands and marshes'
                },
                'virginica': {
                    'description': 'Iris virginica is native to eastern North America.',
                    'characteristics': ['Largest petals', 'Blue to purple flowers', 'Eastern habitats'],
                    'habitat': 'Eastern North America'
                }
            }
            
            predicted_species = list(predictions.values())[0]  # Get first prediction
            info = species_info[predicted_species]
            
            st.markdown(f"""
            **Description**: {info['description']}  
            **Habitat**: {info['habitat']}  
            **Characteristics**: {', '.join(info['characteristics'])}
            """)
        else:
            st.warning("Please train at least one KNN model first in the Classic or Fast KNN tabs!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><b>Iris Flower Classification Mini Project</b> | Classic vs Fast KNN</p>
    <p>Built with ❤️ using Streamlit | Dataset: scikit-learn Iris dataset</p>
</div>
""", unsafe_allow_html=True)