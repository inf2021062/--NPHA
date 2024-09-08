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
from sklearn.feature_selection import SelectKBest, f_classif
from mpl_toolkits.mplot3d import Axes3D
st.title("NPHA")

tabs = ["Data Overview", "Machine Learning", "2D/3D Visualization", "Info"]
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
          method = st.radio ("Algorithms", ("Classification Algorithms", "Feature Selection"))



        
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

          elif method == "Feature Selection":
              st.write("Select the best features")

              X = data.iloc[:, :-1]
              y = data.iloc[:,  -1]

              k = st.slider("Select the number of features", 1, X.shape[1])
              selector = SelectKBest(score_func=f_classif, k=k)
              X_new = selector.fit_transform(X, y)

              st.write("Shape of data after feature selection:", X_new.shape)
              st.write("Selected features scores:")
              st.write(selector.scores_)
             

elif selected_tab == "2D/3D Visualization":
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
        reduction_algorithm = st.selectbox("Dimensonality Reduction", ("PCA 2D", "PCA 3D", "t-SNE"))

        if reduction_algorithm == "PCA 2D":
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(data)
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=X_reduced[:, 0], cmap='viridis')
            plt.colorbar()
            st.pyplot()

        elif reduction_algorithm == "PCA 3D":
            pca = PCA(n_components=3)
            X_reduced = pca.fit_transform(data)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=X_reduced[:, 0], cmap='plasma')
            fig.colorbar(scatter)
            st.pyplot()


        elif reduction_algorithm == "t-SNE":
             tsne = TSNE(n_components=2)
             X_reduced = tsne.fit_transform(data)
             plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=X_reduced [:, 0], cmap='inferno')
             plt.colorbar()
             st.pyplot()

elif selected_tab == "Info":
    st.write("## About Application")
    st.write("This app is design to analyze related data from the research on visits to a doctor by the elderly. It allows users to upload their datasets, perform, using various algorithms, clastering and claasification and visualize the results using dimensionality reduction techniques")

    st.write("## How it works")
    st.write("1. **Data Overview**: Upload your dataset and view its content.")
    st.write("2. **Machine Learning**: Choose and set machine learning algorithms for feature selection and classification.")
    st.write("3. **2D/3D Visualitation**: Visualize your data PCA or t-SNE for dimentionality reduction.")

    st.write("## Development Team")
    st.write("This app developed by a team of students from department of informatics from ionian university.")
    st.write("- **Tilemahos Theodoridis**: Developed the data upload and overview functionality and developed the machine learning algorithm integration.")
    st.write("- **Giorgos Liontos**: Created the 2D/3D visualization features and design the user interface and handled the EDA plots.")

    st.write("## Specific Tasks")
    st.write("- **Tilemahos Theodoridis**: Data upload and overview, PCA and t-SNE integration, also integration of feature selection algorithms.")
    st.write("- **Giorgos Liontos**: Implementation of classification algorithms and accuracy metrics, design and layaout of application, creation of the info tab.")
