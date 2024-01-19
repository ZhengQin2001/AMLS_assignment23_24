from PreprocessingA import Pneu_Preprocessing
import numpy as np
import matplotlib.pyplot as plt

def test_preprocessingB():
    pneu_datapath = "Dataset\pneumoniamnist.npz"
    pneu_data = np.load(pneu_datapath)
    preprocessing_instance = Pneu_Preprocessing(pneu_data, 'train', 'pneumoniamnist')

    # Test _normalize_images
    normalized_images = preprocessing_instance._normalize_images()
    print("Max pixel value:", np.max(normalized_images))
    print("Min pixel value:", np.min(normalized_images))
    print("_________________________________________")

    # Test apply_augmentation
    augmented_data = preprocessing_instance.apply_augmentation(batch_size=32)

    # Get a batch of augmented images and labels
    augmented_images, augmented_labels = next(augmented_data)

    plt.figure(figsize=(12, 12))
    for i in range(9):  # Display the first 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[i].reshape(28, 28), cmap='gray')
        plt.title('Pneumonia' if augmented_labels[i] == 1 else 'Normal')
        plt.axis('off')
    plt.savefig("augmented_image_samples_A.jpg")
    plt.close()

    # Test label distribution
    print("Label Distribution:", preprocessing_instance.label_distribution())

    preprocessing_instance.distribution_plot("label_distribution_Pneu.jpg")

    # Test montage
    preprocessing_instance.montage()

    # Test class_weight
    preprocessing_instance.class_weight()

if __name__ == "__main__":
    test_preprocessingA()
    # test_class_weight()