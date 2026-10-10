from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_KNN(train_data, train_target, test_data, test_target, Hidden, n_neighbors=5):
    print('KNN')

    # Flatten the data if needed
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data)
    X_test = scaler.transform(test_data)

    # Define and train the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, np.argmax(train_target, axis=1))

    # Predict
    pred = knn.predict(X_test)
    pred_one_hot = np.zeros_like(test_target)
    pred_one_hot[np.arange(len(pred)), pred] = 1

    # Evaluate
    Eval = ClassificationEvaluation(test_target, pred_one_hot)
    return Eval
