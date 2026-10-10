import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Classificaltion_Evaluation import ClassificationEvaluation


# Define the Spatio-Temporal Attention Mechanism Layer
class SpatioTemporalAttention(Layer):
    def __init__(self):
        super(SpatioTemporalAttention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[2], input_shape[2]),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1],),
                                 initializer='zeros', trainable=True)

    def call(self, inputs):
        # Perform attention mechanism
        context = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(context, axis=1)  # Attention over time axis (axis=1)
        weighted_sum = tf.reduce_sum(attention_weights * inputs, axis=1)  # Weighted sum of features
        return weighted_sum


# Define the STA-ABi-RNN Model
def Model_STA_ABi_RNN(train_data, train_target, test_data, test_target, KFOLD, sol=None):
    if sol is None:
        sol = [5, 0.01, 1]  # Default solution (LSTM units, epochs, optimizer)

    out, model = RNN_train(train_data, train_target, test_data, test_target, KFOLD, sol)  # RNN training
    pred = out
    min_val = np.min(pred)
    max_val = np.max(pred)
    avg = (min_val + max_val) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0

    Eval = ClassificationEvaluation(pred, test_target)
    return Eval, pred


def RNN_train(trainX, trainY, testX, testY, KFOLD, sol):
    # Reshape the input data for LSTM
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Define the model architecture
    inputs = Input(shape=(1, trainX.shape[2]))  # Input layer
    x = Bidirectional(LSTM(sol[0], return_sequences=True))(inputs)  # Bi-directional LSTM
    x = SpatioTemporalAttention()(x)  # Spatio-temporal attention mechanism
    x = Dense(trainY.shape[1], activation='sigmoid')(x)  # Dense layer for output
    model = Model(inputs, x)  # Final model

    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

    # Train the model
    model.fit(trainX, trainY, epochs=sol[1], batch_size=32, validation_data=(testX, testY), verbose=1)

    # Make predictions
    testPredict = model.predict(testX)
    return testPredict, model
