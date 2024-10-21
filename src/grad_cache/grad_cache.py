from typing import List, Dict, Tuple, Union, Callable, Any, Optional
from itertools import chain
from contextlib import nullcontext
from itertools import repeat
from collections import UserDict
import logging

import torch
from torch import nn, Tensor

from grad_cache.context_managers import RandContext

from accelerate import Accelerator
try:
    from deepspeed.runtime.engine import DeepSpeedEngine
    _is_deepspeed_available = True
except ImportError:
    _is_deepspeed_available = False

logger = logging.getLogger(__name__)


class GradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    def __init__(
            self,
            models: List[nn.Module],
            chunk_sizes: Union[int, List[int]],
            loss_fn: Callable[..., Tensor],
            compute_loss_context_manager,
            accelerator: Accelerator,
            split_input_fn: Optional[Callable[[Any, int], Any]] = None,
            get_rep_fn: Optional[Callable[..., Tensor]] = None
    ):
        """
        Initialize the Gradient Cache class instance.
        :param models: A list of all encoder models to be updated by the current cache.
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss_fn: A loss function that takes arbitrary numbers of representation tensors and
        arbitrary numbers of keyword arguments as input. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param fp16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        """
        self.models = models

        if isinstance(chunk_sizes, int):
            self.chunk_sizes = [chunk_sizes for _ in range(len(models))]
        else:
            self.chunk_sizes = chunk_sizes

        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fn = loss_fn
        self.compute_loss_context_manager = compute_loss_context_manager
        self.accelerator = accelerator

        self._get_input_tensors_strict = False

    def __call__(self, *args, **kwargs):
        """
        Call the cache_step function.
        :return: Current step loss.
        """
        return self.cache_step(*args, **kwargs)
    
    def _split(self, model_input, chunk_size: int) -> List:
        """
        Default split input into chunks.
        """
        # nested dict
        if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, (dict, UserDict)) for x in model_input.values()):
            keys = list(model_input.keys())
            chunked_values: List[List[Dict[str, any]]] = [self.split_inputs(model_input[k], chunk_size=chunk_size) for k in keys]
            return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_values))]

        # dict[str, Tensor]
        if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, Tensor) for x in model_input.values()):
            keys = list(model_input.keys())
            chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
            return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

        # list[Tensor]
        elif isinstance(model_input, list) and all(isinstance(x, Tensor) for x in model_input):
            chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
            return [list(s) for s in zip(*chunked_x)]

        # Tensor
        elif isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        # tuple[list | dict]
        elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
            args_chunks = self.split_inputs(model_input[0], chunk_size)
            kwargs_chunks = self.split_inputs(model_input[1], chunk_size)
            return list(zip(args_chunks, kwargs_chunks))

        else:
            raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

    def split_inputs(self, model_input, chunk_size: int) -> List:
        """
        Split input into chunks. Will call user provided `split_input_fn` if specified. Otherwise,
        it can handle input types of tensor, list of tensors and dictionary of tensors.
        :param model_input: Generic model input.
        :param chunk_size:  Size of each chunk.
        :return: A list of chunked model input.
        """
        # delegate splitting to user provided function
        if self.split_input_fn is not None:
            return self.split_input_fn(model_input, chunk_size)
        
        return self._split(model_input, chunk_size=chunk_size)

    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def model_call(self, model: nn.Module, model_input):
        """
        Literally call the model's __call__ method.
        :param model: model to be called
        :param model_input: input to the model call
        :return: model output
        """
        with self.compute_loss_context_manager():
            if isinstance(model_input, Tensor):
                return model(model_input)
            elif isinstance(model_input, list):
                return model(*model_input)
            elif isinstance(model_input, (dict, UserDict)):
                return model(**model_input)
            elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
                model_args, model_kwargs = model_input
                return model(*model_args, **model_kwargs)
            else:
                raise NotImplementedError

    def get_reps(self, model_out) -> Tensor:
        """
        Return representation tensor from generic model output
        :param model_out: generic model output
        :return: a single tensor corresponding to the model representation output
        """
        if self.get_rep_fn is not None:
            return self.get_rep_fn(model_out)
        else:
            return model_out

    def compute_loss(self, *reps: Tensor, **loss_kwargs) -> Tensor:
        """
        Compute the loss based on the representation tensors. The tensors should be ordered same as the list of models
        registered in this GradCache class instance.
        :param reps: Representations for computing the loss.
        :param loss_kwargs: Keyword arguments input to the loss function.
        :return: the loss tensor.
        """
        loss = self.loss_fn(*reps, **loss_kwargs)
        return loss

    def forward_no_grad(
            self,
            model: nn.Module,
            model_inputs,
    ) -> Tuple[Tensor, List[RandContext]]:
        """
        The first forward pass without gradient computation.
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks.
        :return: A tuple of a) representations and b) recorded random states.
        """
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            for x in model_inputs:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                y = self.model_call(model, x)
                model_reps.append(self.get_reps(y))

        # concatenate all sub-batch representations
        if isinstance(model_reps[0], Tensor):
            model_reps = torch.cat(model_reps, dim=0)
        elif isinstance(model_reps[0], dict):
            keys = set(sum([list(_rep.keys()) for _rep in model_reps], []))
            model_reps = {k: torch.cat([_rep[k] for _rep in model_reps], dim=0) for k in keys}
        else:
            raise TypeError()
        return model_reps, rnd_states

    def build_cache(self, *reps: Tensor | Dict[str, Tensor], **loss_kwargs) -> Tuple[List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        reps_detached = []
        for r in reps:
            if isinstance(r, Tensor):
                reps_detached.append(r.detach().requires_grad_())
            elif isinstance(r, dict):
                reps_detached.append({k: v.detach().requires_grad_() for k, v in r.items()})
            else:
                raise TypeError()

        with self.compute_loss_context_manager():
            loss = self.compute_loss(*reps_detached, **loss_kwargs)

        self.backward_handler(loss, is_last_micro_step=False)

        cache = []
        for r in reps_detached:
            if isinstance(r, Tensor):
                cache.append(r.grad)
            elif isinstance(r, dict):
                cache.append({k: v.grad for k, v in r.items()})
            else:
                raise TypeError()

        return cache, loss.detach()

    def forward_backward(
            self,
            model: nn.Module,
            model_inputs,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            deepspeed_step_triger: bool, # Set to True if you want to perform DeepSpeedEngine.step the last step
            no_sync_except_last: bool = False,
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        :param deepspeed_step_triger: Set to True if you want to perform DeepSpeedEngine.step the last step
        """
        if no_sync_except_last:
            # Only available for DDP/FSDP
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]
        
        # Deepspeed Support
        if deepspeed_step_triger:
            is_last_micro_steps = [False for _ in range(len(model_inputs) - 1)] + [True]
        else:
            is_last_micro_steps = [False for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context, is_last_micro_step in zip(model_inputs, random_states, cached_gradients, sync_contexts, is_last_micro_steps):
            with sync_context():
                with state:
                    y = self.model_call(model, x)
                
                reps = self.get_reps(y)

                if isinstance(reps, dict):
                    # Apply flatten & dot on every values
                    keys = set(chain(reps.keys(), gradient.keys()))
                    surrogate = sum([torch.dot(reps[k].flatten(), gradient[k].flatten()) for k in keys])
                elif isinstance(reps, Tensor) and isinstance(gradient, Tensor):
                    surrogate = torch.dot(reps.flatten(), gradient.flatten())
                else:
                    raise TypeError(f"The type of reps is {type(reps)}, gradient is {type(gradient)}")
                
                self.backward_handler(surrogate, is_last_micro_step=is_last_micro_step)
                

    def backward_handler(self, loss: Tensor, is_last_micro_step: bool):
        """
        Warp the Backward function.
        Args:
            loss: loss for backwards
            is_last_step: Whether this is the last step, this indicator is used for Deepspeed.
        """
        if (hasattr(self.accelerator, 'deepspeed_engine_wrapped')) and \
            (self.accelerator.deepspeed_engine_wrapped is not None) and \
            (not is_last_micro_step):
            # Deepspeed handles micro-batch backward on its own, we need to disable grad accu manually except the last step
            ds_engine: DeepSpeedEngine = self.accelerator.deepspeed_engine_wrapped.engine

            original_is_gradient_accumulation_boundary = ds_engine._is_gradient_accumulation_boundary
            ds_engine.set_gradient_accumulation_boundary(False)
            ds_engine.backward(loss)
            ds_engine.set_gradient_accumulation_boundary(original_is_gradient_accumulation_boundary)

            # # debug
            # print(f"ds_engine.micro_steps = {ds_engine.micro_steps}")
            # print(f"ds_engine.global_steps = {ds_engine.global_steps}")
        else:
            self.accelerator.backward(loss)

    def cache_step(
            self,
            *model_inputs,
            no_sync_except_last: bool = False,
            **loss_kwargs
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        all_reps = []
        all_rnd_states = []

        model_inputs = [self.split_inputs(x, chunk_size) for x, chunk_size in zip(model_inputs, self.chunk_sizes)]

        for model, x in zip(self.models, model_inputs):
            model_reps, rnd_states = self.forward_no_grad(model, x)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        cache, loss = self.build_cache(*all_reps, **loss_kwargs)
        cache = [self._split(x, chunk_size) for x, chunk_size in zip(cache, self.chunk_sizes)]

        deepspeed_step_trigers: List[bool] = [False for _ in range(len(self.models) - 1)] + [True]
        for model, x, model_cache, rnd_states, deepspeed_step_triger in zip(
                self.models, model_inputs, cache, all_rnd_states, deepspeed_step_trigers):
            self.forward_backward(model, x, model_cache, rnd_states, deepspeed_step_triger=deepspeed_step_triger, no_sync_except_last=no_sync_except_last)

        return loss
