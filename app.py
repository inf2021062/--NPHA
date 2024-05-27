import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import MeanShift, AffinityPropagation
st.title("NPHA")

tabs = ["Data Overview", "Machine Learning", "2D Visualization", "Info"]
selected_tab = st.radio("choose tab", tabs )

if selected_tab == "Data Overview":
    uploaded_file = st.file_uploader("upload file in format CSV,TXT (no_header)", type=["csv","txt"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, header=None)
        st.write("Data Preview:")
        st.write(data)

        categorical_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        encoder = OneHotEncoder(sparse_output=False)
        encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
        data = pd.concat([data.drop(categorical_cols, axis=1), encoded_cols], axis=1)


    
elif selected_tab == "Machine Learning":
      uploaded_file = st.file_uploader("upload file in format CSV,TXT (no_header)", type=["csv","txt"])
      if uploaded_file is not None:
          data = pd.read_csv(uploaded_file, header=None)
          st.write("Data Preview:")
          st.write(data)

          categorical_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
          encoder = OneHotEncoder(sparse_output=False)
          encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
          data = pd.concat([data.drop(categorical_cols, axis=1), encoded_cols], axis=1)

          st.write("choose category of the algorithms of Machine Learning")
          method = st.radio ("Algorithms", ("Classification Algorithms", "Clustering Algorithms"))



        
          if method == "Classification Algorithms":
             st.write("Educate Classification Algorithms")

             X = data.iloc[:, :-1]
             y = data.iloc[:, -1]

             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
             rf = RandomForestClassifier() 
             rf.fit(X_train, y_train)

             ml = MLPClassifier()
             ml.fit(X_train, y_train)

             rf_y_pred = rf.predict(X_test)

             ml_y_pred = ml.predict(X_test)

             rf_accuracy = accuracy_score(y_test, rf_y_pred)

             ml_accuracy = accuracy_score(y_test, ml_y_pred)

             st.write(f"the accuracy of the classifier random forest is: {rf_accuracy}")
             st.write(f"the accuracy of the classifier MLP is: {ml_accuracy}")

          elif method == "Clustering Algorithms":
              clustering_algorithm = st.selectbox("choose clustering algorithm", ("Mean Shift", "Affinity Propagation"))

              if clustering_algorithm == "Mean Shift":
                  mean_shift = MeanShift()
                  labels = mean_shift.fit_predict(data)
                  st.write("Cluster labels:")
                  st.write(labels)

              elif clustering_algorithm == "Affinity Propagation":
                  affinity_propagation = AffinityPropagation()
                  labels = affinity_propagation.fit_predict(data)
                  st.write("cluster labels:") 
                  st.write(labels) 


             

elif selected_tab == "2D Visualization":
    uploaded_file = st.file_uploader("upload file in format CSV,TXT (no_header)", type=["csv","txt"])
         
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, header=None)
        st.write("Data Preview:")
        st.write(data)

        categorical_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        encoder = OneHotEncoder(sparse_output=False)
        encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
        data = pd.concat([data.drop(categorical_cols, axis=1), encoded_cols], axis=1)

        st.write("Select Reduction Dimension Algorithm:")
        reduction_algorithm = st.selectbox("Dimensonality Reduction", ("PCA", "t-SNE"))

        if reduction_algorithm == "PCA":
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(data)
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
            st.pyplot()

        elif reduction_algorithm == "t-SNE":
             tsne = TSNE(n_components=2)
             X_reduced = tsne.fit_transform(data)
             plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
             st.pyplot()

elif selected_tab == "Info":
    st.write("## About Application")
    st.write("This app is design to analyze related data from the research on visits to a doctor by the elderly. It allows users to upload their datasets, perform, using various algorithms, clastering and claasification and visualize the results using dimensionality reduction techniques")

    st.write("## How it works")
    st.write("1. **Data Overview**: Upload your dataset and view its content.")
    st.write("2. **Machine Learning**: Choose and set machine learning algorithms for clastering and classification.")
    st.write("3. **2D Visualitation**: Visualize your data PCA or t-SNE for dimentionality reduction.")

    st.write("## Development Team")
    st.write("This app developed by a team of students from department of informatics from ionian university.")
    st.write("- **Tilemahos Theodoridis**: Developed the data upload and overview functionality and developed the machine learning algorithm integration.")
    st.write("- **Giorgos Liontos**: Created the 2D visualization features and design the user interface and handled the EDA plots.")

    st.write("## Specific Tasks")
    st.write("- **Tilemahos Theodoridis**: Data upload and overview, PCA and t-SNE integration, also integration of meanshift and affinity propagation algorithms.")
    st.write("- **Giorgos Liontos**: Implementation of classification algorithms and accuracy metrics, design and layaout of application, creation of the info tab.")





    

