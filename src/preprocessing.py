import kagglehub
import shutil
import os
import glob
import pandas as pd
import numpy as np

def main():
    def good_one_hot(X, n_unique):

        X_copy = X.copy()  
        
        for col in X_copy.select_dtypes(include=['object', 'category']).columns:
            top_categories = X_copy[col].value_counts().nlargest(n_unique).index
            X_copy[col] = X_copy[col].where(X_copy[col].isin(top_categories), other="Other")
        
        X_encoded = pd.get_dummies(X_copy, columns=X_copy.select_dtypes(include=['object', 'category']).columns)
        
        return X_encoded

    def add_interaction_terms(X):
        from itertools import combinations
        interaction_terms = pd.DataFrame(index=X.index)
        continuous_columns = X.select_dtypes(include=[np.number]).columns
        for col1, col2 in combinations(continuous_columns, 2):
            interaction_term_name = f"{col1}_x_{col2}"
            interaction_terms[interaction_term_name] = X[col1] * X[col2]
        return pd.concat([X, interaction_terms], axis=1)
    
    df = pd.read_csv('../data/Time Wasters on Social Media.csv')

    df = good_one_hot(df, 100)

    y = df['ProductivityLoss']
    y = pd.DataFrame({"Brain Rot": y})
    y['Brain Rot'] = y['Brain Rot'].apply(lambda x: 0 if x < 5 else 1)
    # x = df.drop(columns=['ProductivityLoss', 'Satisfaction', 'Addiction Level', 'Self Control'])
    x = df.drop(columns=['ProductivityLoss', 'Satisfaction', 'Addiction Level', 'Self Control'])
    x = add_interaction_terms(x)

    features = x.copy()
    target = y.copy()

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Add these to the data folder for the AutoML portion
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    train.to_csv('./data/train.csv', index=False)
    test.to_csv('./data/test.csv', index=False)
    
if __name__ == '__main__':
    main()