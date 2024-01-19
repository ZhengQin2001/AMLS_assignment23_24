import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PreprocessingA import Pneu_Preprocessing
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from kerastuner import RandomSearch
from sklearn.model_selection import KFold
from keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model


class Model_training:
    def __init__(self, dataset_path="Dataset"):
        pneu_datapath = os.path.join(dataset_path, 'pneumoniamnist.npz')

        # Error handling for loading data
        try:
            pneu_data = np.load(pneu_datapath)
        except FileNotFoundError:
            print(f"Dataset file not found in path: {pneu_datapath}")
            return

        # Initialize preprocessing instances
        self.train = Pneu_Preprocessing(pneu_data, 'train', 'pneumoniamnist')
        self.validation = Pneu_Preprocessing(pneu_data, 'val', 'pneumoniamnist')
        self.test = Pneu_Preprocessing(pneu_data, 'test', 'pneumoniamnist')

        # Extract normalized images and labels
        self.X_train, self.y_train = self.train.normalized_images, self.train.labels
        self.X_val, self.y_val = self.validation.normalized_images, self.validation.labels
        self.X_test, self.y_test = self.test.normalized_images, self.test.labels

    def initial_CNN_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            # Removed the pooling layer here or use a larger stride
            
            Flatten(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy'])
        
        return model

    def CNN_model(self):
        l2_lambda = 0.004
        model = Sequential([
            # Convolutional layer 1
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same', kernel_regularizer=l2(l2_lambda)),
            MaxPooling2D((2, 2)),

            # Convolutional layer 2
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda)),
            MaxPooling2D((2, 2)),

            # Convolutional layer 3
            Conv2D(48, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda)),
            MaxPooling2D((2, 2)),

            # Convolutional layer 4
            Conv2D(48, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda)),
            MaxPooling2D((2, 2)),

            # Flatten and Dropout
            Flatten(),
            Dropout(0.5),
            
            # Dense layer for classification
            Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
        ])

        sgd_optimizer = SGD(learning_rate=0.001, momentum=0.9)

        # Compile the model
        model.compile(optimizer=sgd_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model
    
    def train_and_evaluate(self, early_stopping=True, augment=False, tuned_model=True, save_model=False):
        # Class weights
        class_weights = self.train.class_weight()
        
        if tuned_model:
            model = self.CNN_model()
        else:
            model = self.initial_CNN_model()

        # Set up callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00005)
        callbacks = [EarlyStopping(monitor='val_loss', patience=5), reduce_lr] if early_stopping else [reduce_lr]

        if augment:
            train_data_generator = self.train.apply_augmentation(batch_size=64)
            steps_per_epoch = len(self.X_train) // 64
            model.fit(
                train_data_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=120,
                class_weight=class_weights,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks
            )
        else:
            model.fit(
                self.X_train, self.y_train,
                epochs=50,
                class_weight=class_weights,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks
            )
    
        # Test Set Evaluation
        self.evaluate_model(model)

        if save_model:
            model.summary()
            model.save("TaskA\ModelCNN_pretrained.keras")
            print("Model Saved")

    def evaluate_model(self, model):
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test)
        print("Test accuracy:", test_accuracy)

    def read_model(self, model_file):
        model = load_model(model_file)
        model.summary()
        for layer in model.layers:
            print(layer.get_config())
            model.optimizer.get_config()
        first_layer_weights = model.layers[0].get_weights()

    def _hyperparameter_tuning(self, hp):
        model = Sequential([
            Conv2D(
                filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
                activation='relu',
                input_shape=(28, 28, 1)
            ),
            MaxPooling2D((2, 2)),
            Conv2D(
                filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
                kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
                activation='relu'
            ),
            MaxPooling2D((2, 2)),
            Conv2D(
                filters=hp.Int('conv_3_filter', min_value=32, max_value=64, step=16),
                kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
                activation='relu'
            ),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(
                units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
                activation='relu'
            ),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
    
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        return model

    def build_model(self):
        tuner = RandomSearch(
            self._hyperparameter_tuning,
            objective='val_accuracy',
            max_trials=15,  # Number of trials to run
            executions_per_trial=3,  # Number of models to train for each trial
            directory='output',
            project_name='PneumoniaMNIST'
        )

        tuner.search(self.X_train, self.y_train, epochs=20, validation_data=(self.X_val, self.y_val))

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters()[0]

        print("Best Hyperparameters:", best_hyperparameters.values)

    def train_CNN_model_with_cross_validation(self, num_folds=5, epochs=10, augment=False):
        kfold = KFold(n_splits=num_folds, shuffle=True)
        fold_no = 1
        acc_per_fold = []
        loss_per_fold = []

        for train, test in kfold.split(self.X_train, self.y_train):
            # Define the model architecture
            model = self.CNN_model()  # Assuming create_model is a method that returns a compiled model

            print(f'Training for fold {fold_no} ...')

            # Fit data to model
            if not augment:
                history = model.fit(self.X_train[train], self.y_train[train],
                                    epochs=epochs,
                                    validation_data=(self.X_train[test], self.y_train[test]))
            else:
                train_data_generator = self.train.apply_augmentation(train_indices=train, batch_size=32)
                model.fit(train_data_generator, 
                          steps_per_epoch=len(train) // 32,
                          epochs=epochs, 
                          validation_data=(self.X_train[test], self.y_train[test]))

            # Generate generalization metrics
            scores = model.evaluate(self.X_train[test], self.y_train[test], verbose=0)
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            fold_no += 1

        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

# mti = Model_training_path()
# mti.build_model()
# mti.train_CNN_model_with_cross_validation()