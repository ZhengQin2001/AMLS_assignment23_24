from Model_training import Model_training
from PreprocessingA import Pneu_Preprocessing
import numpy as np

def main():
    pneu_datapath = "Dataset\pneumoniamnist.npz"
    pneu_data = np.load(pneu_datapath)

    print("Data information:")
    train_instance = Pneu_Preprocessing(pneu_data, 'train', 'pneumoniamnist')
    valid_instance = Pneu_Preprocessing(pneu_data, 'val', 'pneumoniamnist')
    test_instance = Pneu_Preprocessing(pneu_data, 'test', 'pneumoniamnist')
    print("> Training: ", train_instance.__len__)
    print("> Validation: ", valid_instance.__len__)
    print("> Test: ", test_instance.__len__)

    # Build model training instance
    model_training_instance = Model_training()

    # Initial model training
    model_training_instance.train_and_evaluate(tuned_model=False)

    # Hyperparameter tuning
    model_training_instance.build_model()

    # Cross_validation
    model_training_instance.train_CNN_model_with_cross_validation()

    # Retrain and evaluate
    model_training_instance.train_and_evaluate()



if __name__ == "__main__":
    main()