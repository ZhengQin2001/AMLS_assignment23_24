import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from MedmnistDataSet import MedmnistDataSet
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

class Pneu_Preprocessing(MedmnistDataSet):
    def __init__(self, datafile, flag, dataset_name):
        super().__init__(datafile, flag, dataset_name)
        
        # Normalize the images
        self.normalized_images = self._normalize_images()

    def _normalize_images(self):
        # Normalize the images to have pixel values in the range [0, 1]
        normal_imgs = self.imgs.astype('float32') / 255.0
        return normal_imgs

    def apply_augmentation(self, batch_size=32):
        # Reshape the images to add a channel dimension
        reshaped_imgs = np.expand_dims(self.imgs, axis=-1)
        # Initialize the data generator for augmentation
        datagen = ImageDataGenerator(
            rotation_range=5,   # Rotation range in degrees
            width_shift_range=0.05,   # Fraction of total width for horizontal shift
            height_shift_range=0.05,  # Fraction of total height for vertical shift
            shear_range=0.05,         # Shear intensity
            rescale = 1./255,        # Normalization
            zoom_range=0.05,          # Zoom range
            horizontal_flip=True,    # Allow horizontal flips
            fill_mode='nearest'      # Strategy for filling new pixels
        )

        # Apply data augmentation to the training images
        if self.flag == 'train':
            # Only apply augmentation to the training data
            datagen.fit(reshaped_imgs)
            return datagen.flow(reshaped_imgs, self.labels, batch_size=batch_size)
        else:
            # For validation and test data, just return the reshaped images and labels
            return (reshaped_imgs, self.labels)

    def label_distribution(self):
        # Count the occurrences of each label
        unique, counts = np.unique(self.labels, return_counts=True)
        label_distribution = dict(zip(unique, counts))
        return label_distribution
    
    def distribution_plot(self, save_path=None):
        # Get the label distribution
        label_dist = self.label_distribution()
        
        # Extract labels and counts for plotting
        labels = list(label_dist.keys())
        counts = list(label_dist.values())

        # Names for each label
        label_names = ['Normal', 'Pneumonia'] if len(labels) == 2 else labels
        
        # Create a bar chart
        plt.bar(label_names, counts, width=0.4)

        # Adding titles and labels
        plt.title('Label Distribution in Dataset')
        plt.xlabel('Labels')
        plt.ylabel('Frequency')

        # Show the plot
        if save_path:
            plt.savefig(save_path, format='png', dpi=300)

    def class_weight(self):
        labels = self.labels.ravel()
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights_dict = dict(enumerate(class_weights))

        return class_weights_dict