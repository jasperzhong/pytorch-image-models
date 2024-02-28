""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
import math
from itertools import islice
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from timm.models import group_parameters

from .adabelief import AdaBelief
from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .adan import Adan
from .lamb import Lamb
from .lars import Lars
from .lion import Lion
from .lookahead import Lookahead
from .madgrad import MADGRAD
from .nadam import Nadam
from .nadamw import NAdamW
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP
from .sgdw import SGDW

_logger = logging.getLogger(__name__)


# optimizers to default to multi-tensor
_DEFAULT_FOREACH = {
    'lion',
}


# START MONKEY PATCHING

def undo_sgd(params: List[Tensor],
             d_p_list: List[Tensor],
             momentum_buffer_list: List[Optional[Tensor]],
             *,
             weight_decay: float,
             momentum: float,
             lr: float,
             dampening: float,
             nesterov: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):
        original_dtype = param.dtype
        param = param.cpu().double()

        d_p = d_p_list[i]
        d_p = d_p.cpu().double()

        if momentum != 0:
            buf = momentum_buffer_list[i]
            buf = buf.cpu().double()

            if nesterov:
                param.add_(d_p.add(buf, alpha=momentum),
                           alpha=lr).div_(1 - lr * weight_decay)
            else:
                param.add_(buf, alpha=lr)

            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            m = torch.empty_like(buf).fill_(momentum)
            buf.add_(d_p, alpha=dampening - 1).div_(m)

            momentum_buffer_list[i].copy_(buf.type(original_dtype))
        else:
            param.add_(d_p, alpha=lr).div_(1 - lr * weight_decay)

        params[i].copy_(param.type(original_dtype))


def undo_adamw(params: List[Tensor],
               grads: List[Tensor],
               exp_avgs: List[Tensor],
               exp_avg_sqs: List[Tensor],
               state_steps: List[int],
               *,
               beta1: float,
               beta2: float,
               lr: float,
               weight_decay: float,
               eps: float):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        # same as step
        # because the worker didn't call optim.adam to update step
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # undo param
        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        param.addcdiv_(exp_avg, denom, value=step_size)
        param.div_(1 - lr * weight_decay)

        # undo Vt
        exp_avg_sq.sub_(grad ** 2, alpha=1 - beta2).div_(beta2)

        # undo Mt
        exp_avg.sub_(grad, alpha=1 - beta1).div_(beta1)


@torch.no_grad()
def sgd_undo(self):
    print("!!!Performing undo!!!")
    for group in self.param_groups:
        params_with_grad = []
        d_p_list = []
        momentum_buffer_list = []
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        undo_sgd(params_with_grad,
                 d_p_list,
                 momentum_buffer_list,
                 weight_decay=weight_decay,
                 momentum=momentum,
                 lr=lr,
                 dampening=dampening,
                 nesterov=nesterov)

        # update momentum_buffers in state
        for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            state = self.state[p]
            state['momentum_buffer'] = momentum_buffer

    print("!!!Undo complete!!!")


@torch.no_grad()
def adamw_undo(self):
    for group in self.param_groups:
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []
        beta1, beta2 = group['betas']

        for p in group['params']:
            if p.grad is not None and p.prev_grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.prev_grad)

                state = self.state[p]
                # Don't need lazy state initialization

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                # record the step after step update
                state_steps.append(state['step'])

        undo_adamw(params_with_grad,
                     grads,
                     exp_avgs,
                     exp_avg_sqs,
                     state_steps,
                     beta1=beta1,
                     beta2=beta2,
                     lr=group['lr'],
                     weight_decay=group['weight_decay'],
                     eps=group['eps'])

        # update exp_avg, exo_avg_sq in state
        for p, mt, vt in zip(params_with_grad, exp_avgs, exp_avg_sqs):
            state = self.state[p]
            state['step'] -= 1
            state['exp_avg_sq'] = vt
            state['exp_avg'] = mt


setattr(optim.SGD, "undo", sgd_undo)
setattr(optim.AdamW, "undo", adamw_undo)
# END MONKEY PATCHING


def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def _group(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def _layer_map(model, layers_per_group=12, num_groups=None):
    def _in_head(n, hp):
        if not hp:
            return True
        elif isinstance(hp, (tuple, list)):
            return any([n.startswith(hpi) for hpi in hp])
        else:
            return n.startswith(hp)

    head_prefix = getattr(model, 'pretrained_cfg', {}).get('classifier', None)
    names_trunk = []
    names_head = []
    for n, _ in model.named_parameters():
        names_head.append(n) if _in_head(
            n, head_prefix) else names_trunk.append(n)

    # group non-head layers
    num_trunk_layers = len(names_trunk)
    if num_groups is not None:
        layers_per_group = -(num_trunk_layers // -num_groups)
    names_trunk = list(_group(names_trunk, layers_per_group))

    num_trunk_groups = len(names_trunk)
    layer_map = {n: i for i, l in enumerate(names_trunk) for n in l}
    layer_map.update({n: num_trunk_groups for n in names_head})
    return layer_map


def param_groups_layer_decay(
        model: nn.Module,
        weight_decay: float = 0.05,
        no_weight_decay_list: Tuple[str] = (),
        layer_decay: float = .75,
        end_layer_decay: Optional[float] = None,
        verbose: bool = False,
):
    """
    Parameter groups for layer-wise lr decay & weight decay
    Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    no_weight_decay_list = set(no_weight_decay_list)
    param_group_names = {}  # NOTE for debugging
    param_groups = {}

    if hasattr(model, 'group_matcher'):
        # FIXME interface needs more work
        layer_map = group_parameters(
            model, model.group_matcher(coarse=False), reverse=True)
    else:
        # fallback
        layer_map = _layer_map(model)
    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(layer_decay ** (layer_max - i)
                        for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = layer_map.get(name, layer_max)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    if verbose:
        import json
        _logger.info("parameter groups: \n%s" %
                     json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        opt=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
    )
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'layer_decay', None) is not None:
        kwargs['layer_decay'] = cfg.layer_decay
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    if getattr(cfg, 'opt_foreach', None) is not None:
        kwargs['foreach'] = cfg.opt_foreach
    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


def create_optimizer_v2(
        model_or_params,
        opt: str = 'sgd',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        foreach: Optional[bool] = None,
        filter_bias_and_bn: bool = True,
        layer_decay: Optional[float] = None,
        param_group_fn: Optional[Callable] = None,
        **kwargs,
):
    """ Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model_or_params (nn.Module): model containing parameters to optimize
        opt: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        foreach: Enable / disable foreach (multi-tensor) operation if True / False. Choose safe default if None
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    if isinstance(model_or_params, nn.Module):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        no_weight_decay = {}
        if hasattr(model_or_params, 'no_weight_decay'):
            no_weight_decay = model_or_params.no_weight_decay()

        if param_group_fn:
            parameters = param_group_fn(model_or_params)
        elif layer_decay is not None:
            parameters = param_groups_layer_decay(
                model_or_params,
                weight_decay=weight_decay,
                layer_decay=layer_decay,
                no_weight_decay_list=no_weight_decay,
            )
            weight_decay = 0.
        elif weight_decay and filter_bias_and_bn:
            parameters = param_groups_weight_decay(
                model_or_params, weight_decay, no_weight_decay)
            weight_decay = 0.
        else:
            parameters = model_or_params.parameters()
    else:
        # iterable of parameters or param groups passed in
        parameters = model_or_params

    opt_lower = opt.lower()
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    if opt_lower.startswith('fused'):
        try:
            from apex.optimizers import (FusedAdam, FusedLAMB, FusedNovoGrad,
                                         FusedSGD)
            has_apex = True
        except ImportError:
            has_apex = False
        assert has_apex and torch.cuda.is_available(
        ), 'APEX and CUDA required for fused optimizers'

    if opt_lower.startswith('bnb'):
        try:
            import bitsandbytes as bnb
            has_bnb = True
        except ImportError:
            has_bnb = False
        assert has_bnb and torch.cuda.is_available(
        ), 'bitsandbytes and CUDA required for bnb optimizers'

    opt_args = dict(weight_decay=weight_decay, **kwargs)

    if lr is not None:
        opt_args.setdefault('lr', lr)

    if foreach is None:
        if opt in _DEFAULT_FOREACH:
            opt_args.setdefault('foreach', True)
    else:
        opt_args['foreach'] = foreach

    # basic SGD & related
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        # NOTE 'sgd' refers to SGD + nesterov momentum for legacy / backwards compat reasons
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum,
                              nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum,
                              nesterov=False, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum,
                         nesterov=True, **opt_args)
    elif opt_lower == 'sgdw' or opt_lower == 'nesterovw':
        # NOTE 'sgd' refers to SGD + nesterov momentum for legacy / backwards compat reasons
        opt_args.pop('eps', None)
        optimizer = SGDW(parameters, momentum=momentum,
                         nesterov=True, **opt_args)
    elif opt_lower == 'momentumw':
        opt_args.pop('eps', None)
        optimizer = SGDW(parameters, momentum=momentum,
                         nesterov=False, **opt_args)

    # adaptive
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'nadam':
        try:
            # NOTE PyTorch >= 1.10 should have native NAdam
            optimizer = optim.Nadam(parameters, **opt_args)
        except AttributeError:
            optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'nadamw':
        optimizer = NAdamW(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamax':
        optimizer = optim.Adamax(parameters, **opt_args)
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'radabelief':
        optimizer = AdaBelief(parameters, rectify=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adagrad':
        opt_args.setdefault('eps', 1e-8)
        optimizer = optim.Adagrad(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adanp':
        optimizer = Adan(parameters, no_prox=False, **opt_args)
    elif opt_lower == 'adanw':
        optimizer = Adan(parameters, no_prox=True, **opt_args)
    elif opt_lower == 'lamb':
        optimizer = Lamb(parameters, **opt_args)
    elif opt_lower == 'lambc':
        optimizer = Lamb(parameters, trust_clip=True, **opt_args)
    elif opt_lower == 'larc':
        optimizer = Lars(parameters, momentum=momentum,
                         trust_clip=True, **opt_args)
    elif opt_lower == 'lars':
        optimizer = Lars(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'nlarc':
        optimizer = Lars(parameters, momentum=momentum,
                         trust_clip=True, nesterov=True, **opt_args)
    elif opt_lower == 'nlars':
        optimizer = Lars(parameters, momentum=momentum,
                         nesterov=True, **opt_args)
    elif opt_lower == 'madgrad':
        optimizer = MADGRAD(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'madgradw':
        optimizer = MADGRAD(parameters, momentum=momentum,
                            decoupled_decay=True, **opt_args)
    elif opt_lower == 'novograd' or opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
            parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9,
                              momentum=momentum, **opt_args)
    elif opt_lower == 'lion':
        opt_args.pop('eps', None)
        optimizer = Lion(parameters, **opt_args)

    # second order
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)

    # NVIDIA fused optimizers, require APEX to be installed
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum,
                             nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=momentum,
                             nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)

    # bitsandbytes optimizers, require bitsandbytes to be installed
    elif opt_lower == 'bnbsgd':
        opt_args.pop('eps', None)
        optimizer = bnb.optim.SGD(
            parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'bnbsgd8bit':
        opt_args.pop('eps', None)
        optimizer = bnb.optim.SGD8bit(
            parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'bnbmomentum':
        opt_args.pop('eps', None)
        optimizer = bnb.optim.SGD(parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'bnbmomentum8bit':
        opt_args.pop('eps', None)
        optimizer = bnb.optim.SGD8bit(
            parameters, momentum=momentum, **opt_args)
    elif opt_lower == 'bnbadam':
        optimizer = bnb.optim.Adam(parameters, **opt_args)
    elif opt_lower == 'bnbadam8bit':
        optimizer = bnb.optim.Adam8bit(parameters, **opt_args)
    elif opt_lower == 'bnbadamw':
        optimizer = bnb.optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'bnbadamw8bit':
        optimizer = bnb.optim.AdamW8bit(parameters, **opt_args)
    elif opt_lower == 'bnblamb':
        optimizer = bnb.optim.LAMB(parameters, **opt_args)
    elif opt_lower == 'bnblamb8bit':
        optimizer = bnb.optim.LAMB8bit(parameters, **opt_args)
    elif opt_lower == 'bnblars':
        optimizer = bnb.optim.LARS(parameters, **opt_args)
    elif opt_lower == 'bnblarsb8bit':
        optimizer = bnb.optim.LAMB8bit(parameters, **opt_args)
    elif opt_lower == 'bnblion':
        optimizer = bnb.optim.Lion(parameters, **opt_args)
    elif opt_lower == 'bnblion8bit':
        optimizer = bnb.optim.Lion8bit(parameters, **opt_args)

    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
