"""REFINE-oriented workflow assembly helpers.

This module does not try to own dataset loading or attack configuration.
Instead, it provides a stable way to assemble the model, UNet, and REFINE
defense objects so later run scripts can stay small and consistent.
"""

from dataclasses import dataclass, field

import torch.nn as nn

from core.defenses import REFINE
from core.models import BaselineMNISTNetwork, ResNet, UNet, UNetLittle
from core.models import vgg as vgg_models
try:
    from torchvision.models import resnet18 as tv_resnet18
    from torchvision.models import resnet50 as tv_resnet50
    from torchvision.models import ResNet18_Weights, ResNet50_Weights
except ImportError:  # pragma: no cover - handled at runtime
    tv_resnet18 = None
    tv_resnet50 = None
    ResNet18_Weights = None
    ResNet50_Weights = None


def _build_resnet18(num_classes, **kwargs):
    return ResNet(18, num_classes=num_classes)


def _build_resnet34(num_classes, **kwargs):
    return ResNet(34, num_classes=num_classes)


def _resolve_torchvision_weights(weights, enum_cls):
    if weights is None or enum_cls is None:
        return weights
    if isinstance(weights, str):
        if weights.upper() == 'DEFAULT':
            return enum_cls.DEFAULT
        return getattr(enum_cls, weights)
    return weights


def _build_torchvision_resnet18(num_classes, weights=None, **kwargs):
    if tv_resnet18 is None:
        raise ImportError('torchvision is required for torchvision-resnet18.')
    weights = _resolve_torchvision_weights(weights, ResNet18_Weights)
    model = tv_resnet18(weights=weights, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_torchvision_resnet50(num_classes, weights=None, **kwargs):
    if tv_resnet50 is None:
        raise ImportError('torchvision is required for torchvision-resnet50.')
    weights = _resolve_torchvision_weights(weights, ResNet50_Weights)
    model = tv_resnet50(weights=weights, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_baseline_mnist(num_classes, **kwargs):
    model = BaselineMNISTNetwork()
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
        if model.classifier.out_features != num_classes:
            raise ValueError('BaselineMNISTNetwork only supports its built-in class count.')
    return model


MODEL_BUILDERS = {
    'resnet18': _build_resnet18,
    'resnet34': _build_resnet34,
    'torchvision-resnet18': _build_torchvision_resnet18,
    'torchvision-resnet50': _build_torchvision_resnet50,
    'baseline-mnist': _build_baseline_mnist,
    'vgg11': lambda num_classes, **kwargs: vgg_models.vgg11(num_classes=num_classes, **kwargs),
    'vgg11_bn': lambda num_classes, **kwargs: vgg_models.vgg11_bn(num_classes=num_classes, **kwargs),
    'vgg13': lambda num_classes, **kwargs: vgg_models.vgg13(num_classes=num_classes, **kwargs),
    'vgg13_bn': lambda num_classes, **kwargs: vgg_models.vgg13_bn(num_classes=num_classes, **kwargs),
    'vgg16': lambda num_classes, **kwargs: vgg_models.vgg16(num_classes=num_classes, **kwargs),
    'vgg16_bn': lambda num_classes, **kwargs: vgg_models.vgg16_bn(num_classes=num_classes, **kwargs),
    'vgg19': lambda num_classes, **kwargs: vgg_models.vgg19(num_classes=num_classes, **kwargs),
    'vgg19_bn': lambda num_classes, **kwargs: vgg_models.vgg19_bn(num_classes=num_classes, **kwargs),
}


UNET_BUILDERS = {
    'unet': lambda in_channels, out_channels, **kwargs: UNet(
        args=None,
        n_channels=in_channels,
        n_classes=out_channels,
        **kwargs,
    ),
    'unet-little': lambda in_channels, out_channels, **kwargs: UNetLittle(
        args=None,
        n_channels=in_channels,
        n_classes=out_channels,
        **kwargs,
    ),
}


@dataclass
class RefinePipelineConfig:
    num_classes: int
    in_channels: int = 3
    model_name: str = 'resnet18'
    model_kwargs: dict = field(default_factory=dict)
    unet_name: str = 'unet-little'
    unet_kwargs: dict = field(default_factory=lambda: {'first_channels': 16})
    refine_kwargs: dict = field(default_factory=dict)


def build_model(model_name, num_classes, **model_kwargs):
    key = model_name.lower()
    if key not in MODEL_BUILDERS:
        supported = ', '.join(sorted(MODEL_BUILDERS))
        raise KeyError(f'Unknown model: {model_name}. Supported: {supported}.')
    return MODEL_BUILDERS[key](num_classes=num_classes, **model_kwargs)


def build_unet(unet_name, in_channels, out_channels, **unet_kwargs):
    key = unet_name.lower()
    if key not in UNET_BUILDERS:
        supported = ', '.join(sorted(UNET_BUILDERS))
        raise KeyError(f'Unknown UNet: {unet_name}. Supported: {supported}.')
    return UNET_BUILDERS[key](in_channels=in_channels, out_channels=out_channels, **unet_kwargs)


def build_refine_defense(config, model=None, unet=None):
    if model is None:
        model = build_model(
            config.model_name,
            num_classes=config.num_classes,
            **config.model_kwargs,
        )

    if unet is None:
        unet = build_unet(
            config.unet_name,
            in_channels=config.in_channels,
            out_channels=config.in_channels,
            **config.unet_kwargs,
        )

    refine_kwargs = dict(config.refine_kwargs)
    refine_kwargs.setdefault('num_classes', config.num_classes)

    return REFINE(
        unet=unet,
        model=model,
        **refine_kwargs,
    )
