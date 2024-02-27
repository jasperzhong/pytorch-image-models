""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch.cuda.amp.grad_scaler import GradScaler, OptState

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from .clip_grad import dispatch_clip_grad

# START MONKEY PATCH


def step(self, optimizer, undo, *args, **kwargs):
    """
    :meth:`step` carries out the following two operations:

    1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
        earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
    2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
        gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

    ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

    Returns the return value of ``optimizer.step(*args, **kwargs)``.

    Args:
        optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
        args:  Any arguments.
        kwargs:  Any keyword arguments.

    .. warning::
        Closure use is not currently supported.
    """
    if (not self._enabled):
        return optimizer.step(*args, **kwargs)

    if "closure" in kwargs:
        raise RuntimeError(
            "Closure use is not currently supported if GradScaler is enabled.")

    self._check_scale_growth_tracker("step")

    optimizer_state = self._per_optimizer_states[id(optimizer)]

    if optimizer_state["stage"] is OptState.STEPPED:
        raise RuntimeError(
            "step() has already been called since the last update().")

    retval = None

    if (hasattr(optimizer, "_step_supports_amp_scaling") and optimizer._step_supports_amp_scaling):
        # This optimizer has customized scale-handling logic, so we can call optimizer.step() directly.
        # The contract with custom optimizers is that their step() should accept an additional,
        # optional grad_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
        # it can query its own state, invoke unscale_ on itself, etc
        retval = optimizer.step(*args, **dict(kwargs, grad_scaler=self))
        optimizer_state["stage"] = OptState.STEPPED
        return retval

    if optimizer_state["stage"] is OptState.READY:
        self.unscale_(optimizer)

    assert len(optimizer_state["found_inf_per_device"]
               ) > 0, "No inf checks were recorded for this optimizer."

    retval = self._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)
    if undo:
        assert hasattr(
            optimizer, "undo"), "The optimizer does not have an undo method"
        optimizer.undo()

        # redo
        retval = self._maybe_opt_step(
            optimizer, optimizer_state, *args, **kwargs)
        print("undo-then-redo finish!")

    optimizer_state["stage"] = OptState.STEPPED

    return retval


GradScaler.step = step

# END MONKEY PATCH


class ApexScaler:
    state_dict_key = "amp"

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if need_update:
            if clip_grad is not None:
                dispatch_clip_grad(amp.master_params(
                    optimizer), clip_grad, mode=clip_mode)
            optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = GradScaler()

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
            undo=False,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if need_update:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer, undo=undo)
            self._scaler.update()

    def perform_undo(self, optimizer):
        self._scaler.undo(optimizer)

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
