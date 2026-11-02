# Copyright (c) ModelScope Contributors. All rights reserved.
import gc
import json
import tempfile
import torch
import weakref
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import swift.megatron.utils.megatron_lm_utils as utils
from swift.utils import torch_utils


class _Temporary:

    def __init__(self):
        self.cycle = self
        self.tensor = torch.ones(16, device='cpu')


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    references = []
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16, device='cpu')
    expected = torch.full_like(model.weight, 0.75)
    optimizer = SimpleNamespace(reloads=0, master=None)

    def reload_model_params():
        optimizer.reloads += 1
        optimizer.master = model.weight.detach().float().clone()

    optimizer.reload_model_params = reload_model_params
    checkpoint_args = SimpleNamespace(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=4, consumed_train_samples=24)

    def generate(*args, **kwargs):
        temporary = _Temporary()
        references.append(weakref.ref(temporary))
        return {'model': {'weight': torch.empty_like(model.weight)}, 'temporary': temporary}

    def load(*args, **kwargs):
        temporary = _Temporary()
        references.append(weakref.ref(temporary))
        return {'args': checkpoint_args, 'iteration': 3, 'model': {'weight': expected.clone()}, 'temporary': temporary}

    with tempfile.TemporaryDirectory() as checkpoint:
        Path(checkpoint, 'latest_checkpointed_iteration.txt').write_text('3')
        args = SimpleNamespace(
            mcore_model=checkpoint,
            tuner_type='full',
            finetune=False,
            no_load_optim=True,
            no_load_rng=True,
            use_megatron_fsdp=False,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            bf16=True,
            fp16=False,
        )
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            with patch.object(utils, 'unwrap_model', return_value=[model]), \
                    patch.object(utils.dist_checkpointing, 'load_common_state_dict',
                                 return_value={'args': checkpoint_args}), \
                    patch.object(utils, '_generate_state_dict', new=generate), \
                    patch.object(utils, '_filter_adapter_state_dict'), \
                    patch.object(utils, 'patch_merge_fn'), \
                    patch.object(utils, 'TorchDistLoadShardedStrategy'), \
                    patch.object(utils, 'FullyParallelLoadStrategyWrapper'), \
                    patch.object(utils.mpu, 'get_data_parallel_group', return_value=None), \
                    patch.object(utils.dist_checkpointing, 'load', new=load), \
                    patch.object(torch.distributed, 'is_initialized', return_value=False), \
                    patch.object(torch_utils, 'is_torch_cuda_available', return_value=False), \
                    patch.object(torch_utils, 'is_torch_npu_available', return_value=False), \
                    patch.object(torch_utils, 'is_torch_musa_available', return_value=False), \
                    patch.object(torch_utils, 'is_torch_mps_available', return_value=False), \
                    patch.object(torch.cuda, 'empty_cache', side_effect=AssertionError('CPU invoked CUDA cache')):
                iteration = utils.load_mcore_checkpoint(args, [model], optimizer=optimizer)
                assert iteration == 3 and args.consumed_train_samples == 24
                assert optimizer.reloads == 1
                assert torch.equal(model.weight, expected)
                assert torch.equal(optimizer.master, model.weight.float())
                assert references and all(reference() is None for reference in references)
        finally:
            if gc_enabled:
                gc.enable()
    result = {
        'passed': True,
        'loader_source': utils.__file__,
        'iteration': iteration,
        'consumed_samples': args.consumed_train_samples,
        'fp32_master_reload_calls': optimizer.reloads,
        'bf16_weights_equal': True,
        'fp32_master_equal': True,
        'checkpoint_temporaries_released': len(references),
        'checkpoint_io_mocked': True,
        'native_loader_and_gc_unmocked': True,
        'gpu_validation': False,
    }
    payload = json.dumps(result, indent=2) + '\n'
    if options.output:
        options.output.write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
