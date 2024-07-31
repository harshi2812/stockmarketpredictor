import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import streamlit as st
import io

# Load and preprocess data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['Day_of_Week'] = df['date'].dt.dayofweek
    df.drop('date', axis=1, inplace=True)
    
    data_new2 = df[['Year', 'Month', 'open', 'high', 'low', 'close']]
    data_new2["tomorrow"] = data_new2["close"].shift(-1)
    data_new2["target"] = data_new2["tomorrow"] > data_new2["close"].astype(float)
    
    return df, data_new2

# Plot data
def plot_data(df):
    st.subheader('Closing Prices Over Time')
    st.line_chart(df['close'])
    
    st.subheader('Distribution of Closing Prices')
    fig, ax = plt.subplots()
    sns.histplot(df['close'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader('Box Plot of Closing Prices')
    fig, ax = plt.subplots()
    sns.boxplot(x=df['close'], ax=ax)
    st.pyplot(fig)

# Plot correlation heatmap
def plot_correlation_heatmap(data):
    correlation_matrix = data.corr()
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Train model and evaluate
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    model = RandomForestClassifier(random_state=1)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = (y_pred == y_test).mean()
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    st.write(f"ROC AUC Score: {roc_auc:.2f}")
    
    return best_model

# Make predictions
def make_predictions(model, new_data):
    predictions = model.predict(new_data)
    return predictions

# Streamlit app
def main():
    st.title('Stock Price Prediction App')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df, data_new2 = load_and_preprocess_data(uploaded_file)
        
        st.write("Data Preview:")
        st.write(df.head())
        
        plot_data(df)
        plot_correlation_heatmap(data_new2)
        
        X = data_new2[['Year', 'Month', 'open', 'high', 'low', 'close']]
        y = data_new2["target"]
        
        best_model = train_model(X, y)
        
        st.subheader('Save Model')
        if st.button('Save Model'):
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            st.write("Model saved successfully!")
        
        st.subheader('Make Predictions')
        st.write("Upload new data for prediction:")
        new_data_file = st.file_uploader("Choose a CSV file", type="csv", key="new_data")
        
        if new_data_file is not None:
            new_data = pd.read_csv(new_data_file)
            new_data['date'] = pd.to_datetime(new_data['date'])
            new_data['Year'] = new_data['date'].dt.year
            new_data['Month'] = new_data['date'].dt.month
            new_data['Day'] = new_data['date'].dt.day
            new_data['Day_of_Week'] = new_data['date'].dt.dayofweek
            new_data.drop('date', axis=1, inplace=True)
            
            new_data = new_data[['Year', 'Month', 'open', 'high', 'low', 'close']]
            predictions = make_predictions(best_model, new_data)
            st.write("Predictions:")
            st.write(predictions)

if __name__ == "__main__":
    main()
