from PreprocessingB import Path_Preprocessing
import numpy as np
import matplotlib.pyplot as plt

def test_preprocessingB():
    path_datapath = "Dataset\pathmnist.npz"
    path_data = np.load(path_datapath)
    path_preprocessing = Path_Preprocessing(path_data, 'train', 'pathmnist')
    
    # Test _normalize_images
    normalized_images = path_preprocessing._normalize_images()
    print("Max pixel value:", np.max(normalized_images))
    print("Min pixel value:", np.min(normalized_images))
    print("_________________________________________")

    # Test apply_augmentation
    augmented_data = path_preprocessing.apply_augmentation(batch_size=32)

    # Get a batch of augmented images and labels
    augmented_images, augmented_labels = next(augmented_data)

    # Create a grid plot for some augmented images
    plt.figure(figsize=(12, 6))
    for i in range(min(len(augmented_images), 10)):  # Display first 10 images
        plt.subplot(2, 5, i + 1)
        image = augmented_images[i]
        plt.imshow(image.squeeze(), cmap='gray')  # Remove the color channel for grayscale images
        plt.title(f'Label: {augmented_labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("augmented_images_samples_B_random.jpg")

    # test montage
    path_preprocessing.montage()
    print("Montage saved")

    # test preprocessing
    print("Label distribution: ", path_preprocessing.label_distribution())
    path_preprocessing.distribution_plot("Label_distribution_Path.jpg")
    print("Class weight: ", path_preprocessing.class_weight())


if __name__ == "__main__":
    test_preprocessingB()