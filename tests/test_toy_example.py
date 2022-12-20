# Imports
# Standard
import os
from pathlib import Path

# 3rd party
import pytest
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import pandas as pd

# CARs
from cars.utils.dataset import generate_mnist_concept_dataset
from cars.utils.plot import plot_global_explanation
from cars.models.mnist import ClassifierMnist
from cars.experiments.mnist import concept_to_class
from cars.explanations.concept import CAR, CAV
from cars.explanations.feature import CARFeatureImportance

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_dir = Path.cwd() / "../src/cars/data/mnist"

# Load concept sets
X_train, C_train = generate_mnist_concept_dataset(
    concept_to_class["Loop"], data_dir, train=True, subset_size=200, random_seed=42
)
X_test, C_test = generate_mnist_concept_dataset(
    concept_to_class["Loop"], data_dir, train=False, subset_size=100, random_seed=42
)

# Load model
model = ClassifierMnist(latent_dim=42).to(device)

# Evaluate latent representation
X_train = torch.from_numpy(X_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
H_train = model.input_to_representation(X_train)
H_test = model.input_to_representation(X_test)

# Fit CAR classifier
car = CAR(device)
car.fit(H_train.detach().cpu().numpy(), C_train)
car.tune_kernel_width(H_train.detach().cpu().numpy(), C_train)

# Fit CAV classifier
cav = CAV(device)
cav.fit(H_train.detach().cpu().numpy(), C_train)

# Predict concept labels
cav_preds = cav.concept_importance(
    H_test.detach().cpu().numpy(),
    torch.from_numpy(C_test.astype(int)),
    10,
    model.representation_to_output,
)
car_preds = car.predict(H_test.detach().cpu().numpy())

# Load test set
test_set = MNIST(data_dir, train=False, download=True)
test_set.transform = transforms.Compose([transforms.ToTensor()])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)


def test_evaluate_feature_importance():
    # Evaluate feature importance on the test set
    baselines = torch.zeros([1, 1, 28, 28]).to(device)  # Black image baseline
    car_features = CARFeatureImportance("Integrated Gradient", car, model, device)
    feature_importance = car_features.attribute(test_loader, baselines=baselines)
    assert isinstance(feature_importance, np.ndarray)
