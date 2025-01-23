import pandas as pd
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.optimizers import Adagrad, Adam
from zenml import step
import tensorflow as tf
from keras._tf_keras.keras.utils import load_img
from keras._tf_keras.keras.models import Sequential, Model
from keras._tf_keras.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input, BatchNormalization
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.utils import plot_model
import time


@step(enable_cache=False)
def compileneuralnetwork(X_train: pd.DataFrame) -> Sequential:
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


@step(enable_cache=False)
def compilecomplexneuralnetwork(X_train: pd.DataFrame) -> Sequential:
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@step(enable_cache=False)
def compilesimpleneuralnetwork(X_train: pd.DataFrame) -> Sequential:
    model = Sequential([
        Dense(32, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


@step(enable_cache=False)
def plotmodel(model: Sequential):
    plot_model(model, to_file='DataPlots/model.png', show_shapes=True)


@step(enable_cache=False)
def trainmodel(model: Sequential, X_train: pd.DataFrame, y_train: pd.Series,
               X_test: pd.DataFrame, y_test: pd.Series):

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
        min_lr=0.0001
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True
    )
    start = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=130,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping],
    )
    end = time.time()
    print(f"Neural Network training took {end - start} seconds")
    return model


@step(enable_cache=False)
def summarizemodel(model: Sequential):
    model.summary()
