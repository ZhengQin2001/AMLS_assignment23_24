from keras.models import load_model
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader

base_path = os.path.dirname(os.path.abspath(__file__))
task_a_path = os.path.join(base_path, 'TaskA')
task_b_path = os.path.join(base_path, 'TaskB')
sys.path.append(task_a_path)
sys.path.append(task_b_path)


from Model_training import Model_training
from Model_training_path import Model_training_path
from Model_training_torch import Model_training_path_torch

def main(train_and_evaluate=False, resnet_result=False):
    model_TA = load_model("TaskA\ModelCNN_pretrained.keras") # Model for Task A
    model_TB1 = load_model("TaskB\DeepCNN_refitted.keras") # Model 1 for Task B
    model_TB2 = load_model("TaskB\DeepCNN_pretrained.keras") # Model 2 for Task B

    print("Task A: ")
    mti = Model_training()
    test_loss, test_accuracy = model_TA.evaluate(mti.X_test, mti.y_test)
    val_loss, val_accuracy = model_TA.evaluate(mti.X_val, mti.y_val)
    print("Test accuracy: ", test_accuracy)
    print("Val accuracy: ", val_accuracy)
    get_error_rate(model_TA, mti.X_test, mti.y_test)

    print("Task B: ")
    mtp = Model_training_path()
    print("Model 1")
    test_loss_B1, test_accuracy_B1 = model_TB1.evaluate(mtp.X_test, mtp.y_test)
    val_loss_B1, val_accuracy_B1 = model_TB1.evaluate(mtp.X_val, mtp.y_val)
    print("Test accuracy: ", test_accuracy_B1)
    print("Val accuracy: ", val_accuracy_B1)
    get_error_rate(model_TB1, mtp.X_test, mtp.y_test)

    print("Model 2")
    test_loss_B2, test_accuracy_B2 = model_TB2.evaluate(mtp.X_test, mtp.y_test)
    val_loss_B2, val_accuracy_B2 = model_TB2.evaluate(mtp.X_val, mtp.y_val)
    print("Test accuracy: ", test_accuracy_B2)
    print("Val accuracy: ", val_accuracy_B2)
    get_error_rate(model_TB2, mtp.X_test, mtp.y_test)

    if train_and_evaluate:
        mti.train_and_evaluate(augment=True)
        mtp.train_and_evaluate(original_model=True, augment=True)
        mtp.train_and_evaluate(original_model=False, augment=True)

    if resnet_result:
        resnet_testing()
        

def get_error_rate(model, X_test, y_test):
    # Get model predictions
    predictions = model.predict(X_test)

    # Calculate Top-1 Accuracy
    top1_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))

    # Calculate Top-5 Accuracy
    top5_correct = np.any(np.argsort(predictions, axis=1)[:, -5:] == np.argmax(y_test, axis=1, keepdims=True), axis=1)
    top5_accuracy = np.mean(top5_correct)

    # Calculate error rates
    top1_error_rate = 1 - top1_accuracy
    top5_error_rate = 1 - top5_accuracy

    print(f"Top-1 Error Rate: {top1_error_rate}")
    print(f"Top-5 Error Rate: {top5_error_rate}")

def resnet_testing():
    mtt = Model_training_path_torch()

    # Instantiate your model
    model = mtt.load_resnet34_model()

    # Load the state dictionary
    state_dict = torch.load('TaskB\resnet34_trained.pth')
    model.load_state_dict(state_dict)

    # Prepare your test dataset and loader
    test_loader = DataLoader(mtt.val_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, top1_preds = outputs.topk(1, 1, True, True)
            _, top5_preds = outputs.topk(5, 1, True, True)

            top1_correct += (top1_preds == target.view(*top1_preds.shape)).sum().item()
            top5_correct += (top5_preds == target.view(-1, 1).expand_as(top5_preds)).sum().item()
            total += target.size(0)

    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    top1_error_rate = 1 - top1_accuracy
    top5_error_rate = 1 - top5_accuracy

    print(f"Test Accuracy: {top1_accuracy * 100}%")
    print(f"Top-1 Error Rate: {top1_error_rate}")
    print(f"Top-5 Error Rate: {top5_error_rate}")

if __name__ == "__main__":
    main(train_and_evaluate=False, resnet_result=True)