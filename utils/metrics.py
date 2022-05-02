import itertools
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from explanations.concept import ConceptExplainer, CAR
from sklearn.metrics import accuracy_score


def evaluate_concept_accuracy(data_loader: DataLoader, concept_explainer: ConceptExplainer, concept_id: int, device: torch.device,
                              model: nn.Module) -> float:
    avg_meter = AverageMeter(name="Accuracy")
    for features, labels, concept_labels in data_loader:
        features = features.to(device)
        representations = model.input_to_representation(features).detach().cpu().numpy()
        predicted_concept_labels = concept_explainer.predict(representations)
        avg_meter.update(accuracy_score(predicted_concept_labels, concept_labels[:, concept_id]), n=len(concept_labels))
    return avg_meter.avg


def perturbation_metric(data_loader: DataLoader, attribution: np.ndarray, device: torch.device, model: nn.Module,
                        concept_explainer: ConceptExplainer, baselines: torch.Tensor, n_perts: list[int]) -> np.ndarray:
    """
    Compute the perturbation sensitivity metric for the specified number of perturbations
    Args:
        data_loader: data loader for the examples that are perturbed
        attribution: feature importance
        model: model for which perturbations are computed
        baselines: baseline features used for perturbation
        n_perts: list containing the number of perturbed features for each perturbation

    Returns:
        a list of perturbation sensitivity for each n_pert in n_perts
    """
    pert_sensitivity = np.empty((len(n_perts), len(data_loader)))
    batch_size = data_loader.batch_size
    for pert_id, n_pert in enumerate(n_perts):
        for batch_id, (batch_features, _) in enumerate(data_loader):
            batch_features = batch_features.to(device)
            batch_shape = batch_features.shape
            flat_batch_shape = batch_features.view(len(batch_features), -1).shape
            batch_attribution = attribution[batch_id*batch_size:batch_id*batch_size+len(batch_features)]
            mask = torch.ones(flat_batch_shape, device=device)
            top_features = torch.topk(torch.abs(torch.from_numpy(batch_attribution)).view(flat_batch_shape), n_pert)[1]
            for k in range(n_pert):
                mask[:, top_features[:, k]] = 0  # Mask the n_pert most important entries
            batch_features_pert = mask*batch_features.view(flat_batch_shape) +\
                                (1-mask)*baselines.view(1, -1)
            batch_features_pert = batch_features_pert.view(batch_shape)
            # Compute the latent shift between perturbed and unperturbed inputs
            batch_reps = model.input_to_representation(batch_features).detach()
            concept_importance = concept_explainer.concept_importance(batch_reps)
            batch_reps_pert = model.input_to_representation(batch_features_pert).detach()
            concept_importance_pert = concept_explainer.concept_importance(batch_reps_pert)
            pert_sensitivity[pert_id, batch_id] = torch.mean(torch.abs(concept_importance-concept_importance_pert)).item()
    return pert_sensitivity


def correlation_matrix(attribution_dic: dict[str, np.ndarray]) -> np.ndarray:
    """
    Computes the correlation matrix between the feature importance methods stored in a dictionary
    Args:
        attribution_dic: dictionary of the form feature_importance_method:feature_importance_scores

    Returns:
        Correlation matrix
    """
    corr_mat = np.empty((len(attribution_dic), len(attribution_dic)))
    for entry_id, (name1, name2) in enumerate(itertools.product(attribution_dic, attribution_dic)):
        corr_mat[entry_id//len(attribution_dic), entry_id%len(attribution_dic)] =\
            np.corrcoef(attribution_dic[name1].flatten(), attribution_dic[name2].flatten())[0, 1]
    return corr_mat


def concept_impact(original_loader: DataLoader, modulated_loader: DataLoader, black_box: nn.Module,
                   car_explainer: CAR, device: torch.device) -> list:
    concept_impacts = []
    for (factual_features, _), [modulated_features] in zip(original_loader, modulated_loader):
        factual_features = factual_features.to(device)
        factual_reps = black_box.input_to_representation(factual_features).detach()
        factual_importance = car_explainer.concept_importance(factual_reps).cpu().numpy()
        modulated_features = modulated_features.to(device).float()
        modulated_reps = black_box.input_to_representation(modulated_features).detach()
        modulated_importance = car_explainer.concept_importance(modulated_reps).cpu().numpy()
        concept_impacts.append(np.mean(np.abs(factual_importance - modulated_importance)))
    return concept_impacts


def modulation_norm(original_loader: DataLoader, modulated_loader: DataLoader, device: torch.device) -> list:
    counterfactual_distances = []
    for (factual_features, _), [modulated_features] in zip(original_loader, modulated_loader):
        factual_features = factual_features.to(device).flatten(1)
        modulated_features = modulated_features.to(device).float().flatten(1)
        counterfactual_distances.append(torch.mean(torch.abs(factual_features-modulated_features)).item())
    return counterfactual_distances


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count