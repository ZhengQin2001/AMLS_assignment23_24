import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# Calculate the absolute path to the TaskA directory
current_dir = os.path.dirname(os.path.abspath(__file__))
taskA_dir = os.path.join(current_dir, '..', 'TaskA')

# Add the TaskA directory to the system path
sys.path.append(taskA_dir)

from PreprocessingA import Pneu_Preprocessing

class Path_Preprocessing(Pneu_Preprocessing):
    def __init__(self, datafile, flag, dataset_name):
        super().__init__(datafile, flag, dataset_name)

    def apply_augmentation_path(self, batch_size=32):
        # Define the augmentation parameters
        datagen = ImageDataGenerator(
            rotation_range=5,      # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.05, # randomly shift images vertically (fraction of total height)
            shear_range=0.05,        # apply shearing transformations
            zoom_range=0.05,         # randomly zoom inside pictures
            horizontal_flip=False,   # randomly flip images horizontally
            fill_mode='nearest'     # fill mode for newly created pixels
        )

        encoded_labels = to_categorical(self.labels, num_classes=9)
        # Apply the augmentation only to the training data
        if self.flag == 'train':
            datagen.fit(self.normalized_images)
            return datagen.flow(self.normalized_images, encoded_labels, batch_size=batch_size)
        else:
            # For validation and test data, return images and labels without augmentation
            return (self.normalized_images, encoded_labels)

    def distribution_plot(self, save_path=None):
        # Get the label distribution
        label_dist = self.label_distribution()
        
        # Extract labels and counts for plotting
        labels = list(label_dist.keys())
        counts = list(label_dist.values())

        # Names for each label
        label_names = ['ADI', 'BACK', 'DEB','LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'] if len(labels) == 9 else labels
        
        # Create a bar chart
        plt.bar(label_names, counts, width=0.4)

        # Adding titles and labels
        plt.title('Label Distribution in Dataset')
        plt.xlabel('Labels')
        plt.ylabel('Frequency')

        # Show the plot
        if save_path:
            plt.savefig(save_path, format='png', dpi=300)

