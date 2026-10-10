import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target, Hidden):
    # Define the DNN model
    model = keras.Sequential([
        layers.Dense(Hidden, activation='relu', input_shape=(Train_Data.shape[1],)),
        layers.Dropout(0.5),
        layers.Dense(Train_Target.shape[1], activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(Train_Target.shape[1], activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Train the model
    train = np.asarray(Train_Data).astype(np.float32)
    test = np.asarray(Test_Data).astype(np.float32)
    model.fit(train, Train_Target, epochs=10, batch_size=32, verbose=1)  # Added epochs and batch_size for training
    pred = model.predict(test)

    # Thresholding predictions for binary classification
    min_val = np.min(pred)
    max_val = np.max(pred)
    avg_val = (min_val + max_val) / 2
    pred[pred >= avg_val] = 1
    pred[pred < avg_val] = 0
    Eval = ClassificationEvaluation(Test_Target, pred)
    return Eval

