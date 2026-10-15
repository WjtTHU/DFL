import torch

# these functions are used for debug


def detect_nan_forward(module, input, output):
    def check_tensor(t, module_name):
        if t is None:
            return
        if torch.is_tensor(t):
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"[Forward] NaN/Inf detected in {module_name}, shape={t.shape}")
        elif isinstance(t, (tuple, list)):
            for i, item in enumerate(t):
                check_tensor(item, f"{module_name}[{i}]")
        elif isinstance(t, dict):
            for k, item in t.items():
                check_tensor(item, f"{module_name}[{k}]")

    check_tensor(output, module.__class__.__name__)


def register_nan_hooks(model):
    for m in model.modules():
        m.register_forward_hook(detect_nan_forward)


def detect_nan_grad(grad, name):
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print(f"[Backward] NaN/Inf detected in grad of {name}")
    return grad


def register_grad_hooks(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(lambda grad, name=name: detect_nan_grad(grad, name))
