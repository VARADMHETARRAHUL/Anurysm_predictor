def main():
    # Import libraries
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
    import pandas as pd

    df = pd.read_csv('aneurysm_train.csv')
    # Keep SeriesInstanceUID aside
    SeriesInstancesUID = df['SeriesInstanceUID']
    df = df.drop(columns=['SeriesInstanceUID'])

    # OneHot encode PatientSex
    ohe_patientsex = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    ohe_patientsex_transform = ohe_patientsex.fit_transform(df[['PatientSex']])

    # Reset index before concat
    df = pd.concat(
        [df.drop(columns=['PatientSex']).reset_index(drop=True),
         ohe_patientsex_transform.reset_index(drop=True)],
        axis=1
    )

    # OneHot encode Modality
    ohe_modality = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    ohe_modality_transform = ohe_modality.fit_transform(df[['Modality']])

    df = pd.concat(
        [df.drop(columns=['Modality']).reset_index(drop=True),
         ohe_modality_transform.reset_index(drop=True)],
        axis=1
    )

    # Split features/labels
    X = df.iloc[:, 0:13]
    y = df.iloc[:, 13]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))


if __name__ == "__main__":
    main()
