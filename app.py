import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import streamlit.components.v1 as components
import sweetviz as sv
from scipy import stats
import scipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score



st.wide = True
# Customize matplotlib and seaborn visualizations
plt.style.use('ggplot')  # For ggplot-like style in matplotlib plots
sns.set(style="whitegrid")  # For seaborn plots with a whitegrid background

def load_data():
    data = pd.read_csv('CC GENERAL.csv')
    return data

data = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to",  ["Preprocessing", "EDA", "KMeans Testing"])


st.markdown('<h1 style="color:black;background-color:White;text-align:center;font-size:50px;">📊 Credit Card Customer Segmentation</h1>', unsafe_allow_html=True)
if page == "Preprocessing":
    st.header(':----------------------------------------------------------:', anchor=None)
    st.markdown('<p style="color:DarkSlateBlue; background-color:skyblue;text-align:center; font-size:20px; padding:10px;">This is a simple web app to view and analyze the credit card data. The dataset contains information about the credit card customers and their spending habits. The dataset can be found [here](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).</p>', unsafe_allow_html=True)
    st.image('process.webp', use_column_width=True)
    st.markdown('<h2 style="background-color:DarkRed; color:White; padding:10px;">🔧 Data preprocessing:</h2>', unsafe_allow_html=True)
    st.image('pre.webp',use_column_width=True)
    st.markdown('<h3 style="background-color:DarkOrange; color:White; padding:10px;">1.1 Initial Data Overview:</h3>', unsafe_allow_html=True)
    st.write('Data shape:', data.shape)
    st.write('Data types in the given dataset is as follows:',data.dtypes)
    st.dataframe(data.head())
    st.write(data.describe().T.style.background_gradient(cmap = 'Spectral'))
    numeric_data = data.select_dtypes(include=[np.number])
    st.write('Correlation Matrix:', numeric_data.corr().T.style.background_gradient(cmap = 'Spectral'))
    st.markdown(data.info(), unsafe_allow_html=False)
    def check_missing_value(data):
        total = data.isnull().sum().sort_values(ascending = False)
        percentage = (data.isnull().sum() / data.isnull().count()*100).sort_values(ascending = True)
        return pd.concat([total, percentage], axis=1, keys = ['total', 'percentage'])

    st.write(check_missing_value(data).style.background_gradient(cmap = 'Spectral'))
    st.markdown('<h3 style="background-color:DarkOrange; color:White; padding:10px;">1.2 Data Cleaning:</h3>', unsafe_allow_html=True)
    st.image('clean.webp',use_column_width=True)
    st.write('Dropping the missing values')
    st.image('missing_values.webp',use_column_width=True)
    data.dropna(inplace=True)
    st.write('Total missing values after cleaning:', data.isnull().sum().sum())
    st.write('Dropping the CUST_ID column')
    data.drop('CUST_ID', axis=1, inplace=True)
    st.markdown('<h3 style="background-color:DarkOrange; color:White; padding:10px;">1.3 Data After Cleaning:</h3>', unsafe_allow_html=True)
    st.dataframe(data.head())
    st.write(data.describe())
    st.markdown('<h3 style="background-color:DarkOrange; color:White; padding:10px;">1.4 Data Analysis Report:</h3>', unsafe_allow_html=True)
    report = sv.analyze(data)
    report.show_html('report.html')
    with open('report.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=800, height=600, scrolling=True)
elif page == "EDA":
    st.markdown('<h1 style="background-color:MidnightBlue; color:White; padding:10px;">🔍 Exploratory Data Analysis:</h1>', unsafe_allow_html=True)
    st.image('eda1.gif',use_column_width=True)
    
    def feat_plot(feature):
        st.markdown(f'<h2 style="background-color:DarkSlateBlue; color:White; padding:10px;">📊 Analysis for {feature.name}</h2>', unsafe_allow_html=True)
        feature = feature.dropna()

        fig, axs = plt.subplots(1, 3, figsize=(16, 6))
        
        feature.plot(kind = 'hist', ax=axs[0], color='LightSteelBlue', edgecolor='black')
        axs[0].set_title(f'{feature.name} Histogram', color='MidnightBlue')
        
        if len(feature.unique()) > 1:
            mu, sigma = scipy.stats.norm.fit(feature)
            sns.histplot(feature, kde=True, ax=axs[1], color='LightSteelBlue') 
            axs[1].axvline(mu, linestyle = '--', color = 'DarkSlateBlue')
            axs[1].axvline(mu + sigma, linestyle = '--', color = 'DarkSlateBlue')
            axs[1].axvline(mu - sigma, linestyle = '--', color = 'DarkSlateBlue')
            axs[1].set_title(f'{feature.name} Distribution', color='MidnightBlue')
        else:
            axs[1].text(0.5, 0.5, 'Not enough unique values to plot distribution', ha='center', va='center', color='DarkSlateBlue')
        
        feature.plot(kind = 'box', ax=axs[2], color='LightSteelBlue')
        axs[2].set_title(f'{feature.name} Boxplot', color='MidnightBlue')
        
        st.pyplot(fig)
    num_feature = data.select_dtypes(exclude='object')
    for i, column in enumerate(num_feature.columns):
        st.markdown(f'<h3 style="background-color:SteelBlue; color:White; padding:10px;">{i+1}. {column}</h3>', unsafe_allow_html=True)
        feat_plot(num_feature[column])

    def show_regplot(data, x, y):
        st.markdown(f'<h2 style="background-color:DarkSlateBlue; color:White; padding:10px;">📈 Regression plot of {y} vs {x}</h2>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.regplot(data = data, y = y, x = x, ax=ax, color='LightSteelBlue')
        ax.set_title(f'Regression plot of {y} vs {x}', color='DarkSlateBlue')
        st.pyplot(fig)

    show_regplot(data, 'CREDIT_LIMIT', 'PAYMENTS')
    show_regplot(data, 'BALANCE_FREQUENCY', 'BALANCE')

    num_feature = data.select_dtypes(include=[np.number])
    def show_heatmap(num_feature):
        st.markdown('<h2 style="background-color:DarkSlateBlue; color:White; padding:10px;">🔥 Heatmap of Feature Correlations</h2>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.heatmap(num_feature.corr(), annot = True, cmap = 'coolwarm', ax=ax)
        ax.set_title('Heatmap of Feature Correlations', color='MidnightBlue')
        st.pyplot(fig)

    show_heatmap(num_feature)
elif page == "KMeans Testing":
    st.markdown('<h1 style="background-color:DarkSlateBlue; color:White; padding:10px;">KMeans Testing</h1>', unsafe_allow_html=True)
    st.title(':-----------------------------------------------:')
    st.markdown('<h2 style="background-color:DarkSlateBlue; color:White; padding:10px;">🔬 KMeans Clustering</h2>', unsafe_allow_html=True)

    # Exclude 'CUST_ID' column
    data_to_scale = data.drop('CUST_ID', axis=1)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_to_scale)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_scaled)

    # Determine the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_imputed)
        wcss.append(kmeans.inertia_)
    st.title(' ')
    st.markdown('<h2 style="background-color:DarkSlateBlue; color:White; padding:10px;">Elbow Method Graph</h2>', unsafe_allow_html=True)
    # Plot the elbow method graph
    st.empty()
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss)
    st.image('elbow.png',use_column_width=True)
    st.empty()
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)
    st.empty()  # Add an empty space

    # Apply KMeans to the data
    optimal_clusters = st.slider('Select the number of clusters', 2, 10, 5)
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(data_imputed)

    # Calculate silhouette score
    silhouette = silhouette_score(data_imputed, pred_y)
    st.write('Silhouette Score: ', silhouette)

    st.markdown('<h2 style="background-color:DarkSlateBlue; color:White; padding:10px;">Clusters of Customers</h2>', unsafe_allow_html=True)
    # Visualize the clusters
    st.empty()  # Add an empty space
    fig, ax = plt.subplots()
    ax.scatter(data_imputed[pred_y == 0, 0], data_imputed[pred_y == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    ax.scatter(data_imputed[pred_y == 1, 0], data_imputed[pred_y == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    ax.set_title('Clusters of customers')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.legend()
    st.pyplot(fig)

    # Implementing K-Means
    kmeans = KMeans(n_clusters=4, random_state=32, max_iter=500)
    y_kmeans = kmeans.fit_predict(data_imputed)
    st.markdown('<h2 style="background-color:DarkSlateBlue; color:White; padding:10px;">K-Means Visualizer & Plots</h2>', unsafe_allow_html=True)
    # Define K-Means Visualizer & Plots
    st.empty()  # Add an empty space
    def visualizer(kmeans, y_kmeans, data_imputed):
        # --- Figures Settings ---
        cluster_colors=['#FFBB00', '#3C096C', '#9D4EDD', '#FFE270']
        labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Centroids']
        title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
        text_style=dict(fontweight='bold', fontfamily='serif')
        scatter_style=dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
        legend_style=dict(borderpad=2, frameon=False, fontsize=8)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # --- Silhouette Plots ---
        s_viz = SilhouetteVisualizer(kmeans, ax=ax1, colors=cluster_colors)
        s_viz.fit(data_imputed)
        s_viz.finalize()
        s_viz.ax.set_title('Silhouette Plots of Clusters\n', **title)
        s_viz.ax.tick_params(labelsize=7)
        for text in s_viz.ax.legend_.texts:
            text.set_fontsize(9)
        for spine in s_viz.ax.spines.values():
            spine.set_color('None')
        s_viz.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), **legend_style)
        s_viz.ax.grid(axis='x', alpha=0.5, color='#9B9A9C', linestyle='dotted')
        s_viz.ax.grid(axis='y', alpha=0)
        s_viz.ax.set_xlabel('\nCoefficient Values', fontsize=9, **text_style)
        s_viz.ax.set_ylabel('Cluster Labels\n', fontsize=9, **text_style)
            
        # --- Clusters Distribution ---
        y_kmeans_labels = list(set(y_kmeans.tolist()))
        for i in y_kmeans_labels:
            ax2.scatter(data_imputed[y_kmeans==i, 0], data_imputed[y_kmeans == i, 1], s=50, c=cluster_colors[i], **scatter_style)
        ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=65, c='#0353A4', label='Centroids', **scatter_style)
        for spine in ax2.spines.values():
            spine.set_color('None')
        ax2.set_title('Scatter Plot Clusters Distributions\n', **title)
        ax2.legend(labels, bbox_to_anchor=(0.95, -0.05), ncol=5, **legend_style)
        ax2.grid(axis='both', alpha=0.5, color='#9B9A9C', linestyle='dotted')
        ax2.tick_params(left=False, right=False , labelleft=False , labelbottom=False, bottom=False)
        ax2.spines['bottom'].set_visible(True)
        ax2.spines['bottom'].set_color('#CAC9CD')
        
        # --- Bar Chart ---
        unique, counts = np.unique(y_kmeans, return_counts=True)
        df_bar = dict(zip(unique, counts))
        total = sum(df_bar.values())
        bar_label = {key: round(value/total*100, 2) for key, value in df_bar.items()}

        ax3.set_title('Percentage of Each Clusters\n', **title)
        ax3.set_aspect(aspect='auto')
        ax3.bar(range(len(bar_label)), list(bar_label.values()), color=cluster_colors)
        ax3.set_xticks(range(len(bar_label)))
        ax3.set_xticklabels([f"Cluster {i+1}" for i in range(len(bar_label))])
        for i, v in enumerate(bar_label.values()):
            ax3.text(i, v + 3, f"{v}%", ha='center', color='black', fontweight='bold')
        ax3.set_xlabel('Clusters', fontsize=9, **text_style)
        ax3.set_ylabel('Percentage (%)', fontsize=9, **text_style)

        # --- Suptitle & WM ---
        plt.suptitle('Credit Card Customer Clustering using K-Means\n', fontsize=14, **text_style)
        plt.gcf().text(0.9, 0.03, 'kaggle.com/caesarmario', style='italic', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)  # Pass the figure to st.pyplot()
    visualizer(kmeans, y_kmeans, data_imputed)
    # --- Evaluate Clustering Quality Function ---
    def evaluate_clustering(X, y):
        db_index = round(davies_bouldin_score(X, y), 3)
        s_score = round(silhouette_score(X, y), 3)
        ch_index = round(calinski_harabasz_score(X, y), 3)
        st.markdown("<h1 style='text-align: center; background-color:DarkSlateBlue; color:White'>.: Evaluate Clustering Quality :.</h1>", unsafe_allow_html=True)
        st.title('')
        st.markdown("<p style='text-align: center; color: white;background-color:dark'>.: Davies-Bouldin Index: <b>{}</b></p>".format(db_index), unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: white;background-color:dark'>.: Silhouette Score: <b>{}</b></p>".format(s_score), unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: white;background-color:dark'>.: Calinski Harabasz Index: <b>{}</b></p>".format(ch_index), unsafe_allow_html=True)
        return db_index, s_score, ch_index

    # --- Evaluate K-Means Cluster Quality ---
    db_kmeans, ss_kmeans, ch_kmeans = evaluate_clustering(data_imputed, pred_y)