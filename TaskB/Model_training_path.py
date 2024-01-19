from PreprocessingB import Path_Preprocessing
import tensorflow as tf
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, Add, Input
from tensorflow.keras.models import Model
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from keras.utils import to_categorical
from keras.regularizers import l2, l1_l2
import optuna
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

best_accuracy = 0.8547620868682861  # Global variable to store the best accuracy
best_model_path = 'TaskB\DeepCNN_pretrained.keras'  # Path to save the best model

class Model_training_path:
    def __init__(self, dataset_path="Dataset"):
        path_datapath = os.path.join(dataset_path, 'pathmnist.npz')

        # Error handling for loading data
        try:
            path_data = np.load(path_datapath)
        except FileNotFoundError:
            print(f"Dataset file not found in path: {path_datapath}")
            return

        # Initialize preprocessing instances
        self.train = Path_Preprocessing(path_data, 'train', 'pathmnist')
        self.validation = Path_Preprocessing(path_data, 'val', 'pathmnist')
        self.test = Path_Preprocessing(path_data, 'test', 'pathmnist')

        # Extract normalized images and labels
        self.X_train, self.y_train = self.train.normalized_images, self.train.labels
        self.X_val, self.y_val = self.validation.normalized_images, self.validation.labels
        self.X_test, self.y_test = self.test.normalized_images, self.test.labels

        # One-hot encode the labels
        self.y_train = to_categorical(self.y_train, num_classes=9)
        self.y_val = to_categorical(self.y_val, num_classes=9)
        self.y_test = to_categorical(self.y_test, num_classes=9)
        self.input_shape = (28, 28, 3)
        self.num_classes = 9

    def deep_CNN_model(self):
        # Model configuration
        l2_reg = 0.009935273788869381  # L2 regularization factor
        dropout_rate = 0.19338599188875222  # Dropout rate
        learning_rate = 1e-3

        model = Sequential([
            # Conv Block 1
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=self.input_shape, kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_rate),

            # Conv Block 2
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_rate),

            # Flatten and Dense layers
            Flatten(),
            Dense(51, activation='relu', kernel_regularizer=l2(l2_reg)),  # Intermediate dense layer
            Dropout(dropout_rate),
            Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(l2_reg))  # Output layer
        ])

        # Model summary
        model.summary()
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        return model


    def deep_CNN_model2(self):
        learning_rate = 0.0001
        l2_reg = 0.00405126390978694
        
        model = Sequential([
            # First convolutional layer
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=self.input_shape, kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            # Second convolutional layer
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            # Third convolutional layer
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(0.33176614334913945),

            # Flattening the layers
            Flatten(),

            # First fully-connected layer
            Dense(128, activation='relu'),
            Dropout(0.5),

            # Second fully-connected layer
            Dense(64, activation='relu'),
            Dropout(0.5),

            # Output layer
            Dense(9, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        # Model summary
        model.summary()

        return model


    def build_model(self, trial):
        num_conv_blocks = trial.suggest_int('num_conv_blocks', 2, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.4)

        # Hyperparameters to be tuned
        num_dense_units = trial.suggest_int('num_dense_units', 100, 150)
        l2_lambda = trial.suggest_float('l2_lambda', 1e-3, 1e-2)
        learning_rate = 1e-3

        model = Sequential()

        for i in range(num_conv_blocks):
            num_filters = trial.suggest_categorical('num_filters_{}'.format(i), [32, 64])
            if i == 0:
                # Only the first layer receives the input_shape argument
                model.add(Conv2D(num_filters, kernel_size=(3, 3), activation='relu', 
                                input_shape=self.input_shape, padding='same', 
                                kernel_regularizer=l2(l2_lambda)))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))
            else:
                # Subsequent layers do not need the input_shape
                model.add(Conv2D(num_filters, kernel_size=(3, 3), activation='relu', 
                                padding='same', kernel_regularizer=l2(l2_lambda)))
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        model.add(Flatten())
        model.add(Dense(num_dense_units, activation='relu', kernel_regularizer=l2(l2_lambda)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(l2_lambda)))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def build_model2(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 5e-05, 0.00015)
        l2_reg = trial.suggest_float('l2_reg', 0.001, 0.01)
        
        model = Sequential([
            # First convolutional layer
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=self.input_shape, kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            # Second convolutional layer
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Third convolutional layer
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(trial.suggest_float("dropout_rate", 0.2, 0.5)),

            # Flattening the layers
            Flatten(),

            # First fully-connected layer
            Dense(128, activation='relu'),
            Dropout(0.5),

            # Second fully-connected layer
            Dense(64, activation='relu'),
            Dropout(0.5),

            # Output layer
            Dense(9, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def optimize_hyperparameters(self, n_trials=300, early_stopping=True):
        def objective(trial):
            model = self.build_model2(trial)

            # Callbacks
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
            callbacks = [EarlyStopping(monitor='val_loss', patience=5), reduce_lr] if early_stopping else [reduce_lr]

            # Class weights
            class_weights = self.train.class_weight()

            train_data_generator = self.train.apply_augmentation_path(batch_size=32)

            # Fit the model
            model.fit(
                train_data_generator, 
                validation_data=(self.X_val, self.y_val), 
                class_weight=class_weights, 
                callbacks=callbacks,
                steps_per_epoch=len(self.X_train) // 32, # Number of steps per epoch
                epochs=30, 
                verbose=0
                )

            # Evaluate the model on the validation set
            loss, accuracy = model.evaluate(self.X_val, self.y_val, verbose=0)

            # Check if the current model is the best one
            global best_accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model.save(best_model_path)  # Save the best model
                print(f'Best model saved to {best_model_path}')

            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Retrieve the best hyperparameters
        best_hyperparams = study.best_trial.params
        print(f"Best hyperparameters: {best_hyperparams}")

        # Rebuild and return the best model
        best_model = self.build_model(study.best_trial)
        return best_model

    def read_model(self, model_file):
        model = load_model(model_file)
        model.summary()
        for layer in model.layers:
            print(layer.get_config())
            model.optimizer.get_config()
        first_layer_weights = model.layers[0].get_weights()


    def evaluate(self, model):
        # Test Set Evaluation
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test)
        if test_accuracy > 0.85:
            model.save("TaskB\DeepCNN_augment.keras")
            print("Model saved.")
        print("Test accuracy:", test_accuracy)


    def train_and_evaluate(self, original_model=False, early_stopping=True, augment=False):
        # Class weights
        class_weights = self.train.class_weight()

        # Load the best model
        if original_model:
            model = self.deep_CNN_model()
        else:
            model = self.deep_CNN_model2()

        # Set up callbacks
        # Create the ReduceLROnPlateau callback
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)

        callbacks = [EarlyStopping(monitor='val_loss', patience=5), reduce_lr] if early_stopping else [reduce_lr]

        if augment:
            # Set up data augmentation using the apply_augmentation method of the Path_Preprocessing class

            train_data_generator = self.train.apply_augmentation_path(batch_size=64)
            # Fit the model using a generator
            model.fit(
                train_data_generator,
                steps_per_epoch=len(self.X_train) // 64, # Number of steps per epoch
                epochs=30,
                batch_size=8,
                class_weight=class_weights,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks
            )
        else:
            model.fit(
                self.X_train, self.y_train,
                epochs=30,
                class_weight=class_weights,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks
            )

        # Test Set Evaluation
        self.evaluate(model)

# mtp = Model_training_path()
# # mtp.optimize_hyperparameters()