# Copyright (c) ModelScope Contributors. All rights reserved.
import ast
import gc
import os
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def _load_functions(path, names, namespace):
    # Exercise the real loader without importing the optional Megatron/CUDA dependency stack.
    source = ast.parse(path.read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f'Missing functions in {path}: {names}')
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)


class _CheckpointTemporary:

    def __init__(self):
        self.cycle = self


class TestCheckpointLoadMemory(unittest.TestCase):

    def setUp(self):
        gc_enabled = gc.isenabled()
        gc.disable()
        self.addCleanup(gc.enable if gc_enabled else gc.disable)
        self.events = []
        self.references = []
        self.gc_checks = 0
        self.generated_args = None
        self.generated_kwargs = None
        self.filtered_args = []
        self.args = SimpleNamespace(
            mcore_model='checkpoint',
            mcore_adapter='adapter',
            no_load_optim=False,
            no_load_rng=False,
            finetune=False,
            tuner_type='full',
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            use_distributed_optimizer=True,
            use_megatron_fsdp=False,
            data_parallel_random_init=False,
            fp16=False,
            bf16=True,
            consumed_train_samples=0,
        )
        self.checkpoint_args = SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            no_save_optim=False,
            no_save_rng=False,
            use_megatron_fsdp=True,
            consumed_train_samples=24,
        )
        self.rng = {
            'random_rng_state': 'python',
            'np_rng_state': 'numpy',
            'torch_rng_state': 'torch',
            'cuda_rng_state': 'cuda',
            'rng_tracker_states': 'tracker',
        }
        self.models = [SimpleNamespace(load_state_dict=self._load_model)]
        self.optimizer = SimpleNamespace(
            load_state_dict=Mock(side_effect=lambda state: self.events.append(('optimizer', state))),
            reload_model_params=Mock(side_effect=lambda: self.events.append(('reload', ))),
        )
        self.scheduler = SimpleNamespace(
            load_state_dict=Mock(side_effect=lambda state: self.events.append(('scheduler', state))))
        self.cuda = SimpleNamespace(empty_cache=Mock(side_effect=self._empty_cache), set_rng_state=Mock())
        self.tracker = SimpleNamespace(set_states=Mock())
        self.namespace = {
            'os':
            os,
            'gc':
            gc,
            'torch':
            SimpleNamespace(
                cuda=self.cuda,
                set_rng_state=Mock(),
                distributed=SimpleNamespace(
                    is_initialized=lambda: True, barrier=lambda: self.events.append(('barrier', )))),
            'is_torch_cuda_available':
            lambda: True,
            'is_torch_npu_available':
            lambda: False,
            'is_torch_musa_available':
            lambda: False,
            'is_torch_mps_available':
            lambda: False,
            'random':
            SimpleNamespace(setstate=Mock()),
            'np':
            SimpleNamespace(random=SimpleNamespace(set_state=Mock())),
            'tensor_parallel':
            SimpleNamespace(get_cuda_rng_tracker=lambda: self.tracker),
            'logger':
            SimpleNamespace(info=Mock()),
            'unwrap_model':
            lambda models: models,
            '_load_iteration':
            lambda path: 3,
            'get_sharded_sd_metadata':
            lambda args: {
                'distrib_optim_sharding_type': 'fsdp_dtensor'
            },
            '_get_rng_state':
            lambda **kwargs: self._temporary(),
            '_generate_state_dict':
            self._generate,
            '_filter_adapter_state_dict':
            lambda state, peft, adapter_name: self.filtered_args.append((peft, adapter_name)),
            'patch_merge_fn':
            lambda state: None,
            '_load_optimizer_state_dict':
            lambda optimizer, state: optimizer.load_state_dict(state),
            'dist_checkpointing':
            SimpleNamespace(load_common_state_dict=self._common, load=self._load),
            'fsdp_checkpoint':
            SimpleNamespace(
                is_checkpoint=lambda path: True,
                load_common_state_dict=self._common,
                load_checkpoint=lambda args, state, model, path: self._load(state, path),
                select_rng_state=lambda states, random_init: states[0]),
            'mcore_017':
            True,
            'TorchDistLoadShardedStrategy':
            self._temporary,
            'FullyParallelLoadStrategyWrapper':
            lambda strategy, group: strategy,
            'mpu':
            SimpleNamespace(get_data_parallel_group=lambda **kwargs: None, get_data_parallel_rank=lambda: 0),
        }
        root = Path(__file__).resolve().parents[2]
        _load_functions(root / 'swift/utils/torch_utils.py', ('gc_collect', 'empty_cache'), self.namespace)
        self.real_gc_collect = self.namespace['gc_collect']
        self.namespace['gc_collect'] = self._collect
        _load_functions(root / 'swift/megatron/utils/megatron_lm_utils.py',
                        ('load_mcore_checkpoint', '_load_mcore_checkpoint'), self.namespace)

    def _temporary(self):
        temporary = _CheckpointTemporary()
        self.references.append(weakref.ref(temporary))
        return temporary

    def _common(self, path):
        return {'args': self.checkpoint_args, 'temporary': self._temporary()}

    def _generate(self, args, models, optimizer, scheduler, rng, **kwargs):
        self.generated_args = (optimizer, scheduler, rng is not None)
        self.generated_kwargs = kwargs
        return {'model': {'temporary': self._temporary()}, 'rng_state': rng}

    def _load(self, state, path, strategy=None):
        self.events.append(('load', path))
        loaded = {
            'args': self.checkpoint_args,
            'optimizer': {
                'step': 3
            },
            'opt_param_scheduler': {
                'num_steps': 24
            },
            'rng_state': [self.rng],
            'temporary': self._temporary(),
        }
        for index in range(len(self.models)):
            key = 'model' if len(self.models) == 1 else f'model{index}'
            loaded[key] = {'weight': index + 7, 'temporary': self._temporary()}
        return loaded

    def _load_model(self, state, **kwargs):
        self.events.append(('model', state['weight']))

    def _empty_cache(self):
        self.assertTrue(self.references)
        self.assertTrue(all(reference() is None for reference in self.references), 'checkpoint temporaries still alive')
        self.events.append(('empty_cache', ))

    def _collect(self):
        self.assertTrue(any(reference() is not None for reference in self.references))
        self.gc_checks += 1
        self.real_gc_collect()

    def _run_load(self, **kwargs):
        return self.namespace['load_mcore_checkpoint'](self.args, self.models, self.optimizer, self.scheduler, **kwargs)

    def test_resume_releases_temporaries_after_optimizer_rng_and_barrier(self):
        self.assertEqual(self._run_load(), 3)
        self.assertEqual(self.args.consumed_train_samples, 24)
        self.assertEqual(self.events, [
            ('load', 'checkpoint/iter_0000003'),
            ('model', 7),
            ('optimizer', {
                'step': 3
            }),
            ('scheduler', {
                'num_steps': 24
            }),
            ('barrier', ),
            ('empty_cache', ),
        ])
        self.optimizer.reload_model_params.assert_not_called()
        self.namespace['random'].setstate.assert_called_once_with('python')
        self.namespace['np'].random.set_state.assert_called_once_with('numpy')
        self.namespace['torch'].set_rng_state.assert_called_once_with('torch')
        self.cuda.set_rng_state.assert_called_once_with('cuda')
        self.tracker.set_states.assert_called_once_with('tracker')
        self.assertEqual(self.gc_checks, 1)

    def test_model_only_resume_reloads_fp32_master_and_preserves_iteration(self):
        self.args.no_load_optim = self.args.no_load_rng = True
        self.assertEqual(self._run_load(), 3)
        self.assertEqual(self.args.consumed_train_samples, 24)
        self.optimizer.reload_model_params.assert_called_once_with()
        self.optimizer.load_state_dict.assert_not_called()
        self.scheduler.load_state_dict.assert_not_called()
        self.cuda.set_rng_state.assert_not_called()
        self.assertEqual(self.generated_args, (None, None, False))
        self.assertEqual(self.events[-3:], [('reload', ), ('barrier', ), ('empty_cache', )])

    def test_finetune_resets_iteration_without_restoring_sample_counter(self):
        self.args.finetune = True
        self.args.consumed_train_samples = 16
        self.assertEqual(self._run_load(), 0)
        self.assertEqual(self.args.consumed_train_samples, 16)
        self.optimizer.reload_model_params.assert_called_once_with()
        self.optimizer.load_state_dict.assert_not_called()
        self.scheduler.load_state_dict.assert_not_called()
        self.cuda.set_rng_state.assert_not_called()

    def test_cpu_collects_temporaries_without_calling_cuda_allocator(self):
        self.namespace['is_torch_cuda_available'] = lambda: False
        self.assertEqual(self._run_load(), 3)
        self.cuda.empty_cache.assert_not_called()
        self.assertTrue(all(reference() is None for reference in self.references))
        self.assertEqual(self.gc_checks, 1)

    def test_fsdp_load_preserves_optimizer_and_rng_semantics(self):
        self.args.use_megatron_fsdp = True
        self.assertEqual(self._run_load(), 3)
        self.optimizer.load_state_dict.assert_called_once_with({'step': 3})
        self.cuda.set_rng_state.assert_called_once_with('cuda')
        self.assertTrue(self.generated_kwargs['fsdp_dtensor'])
        self.cuda.empty_cache.assert_called_once_with()

    def test_virtual_pipeline_loads_all_chunks_before_cleanup(self):
        self.models.append(SimpleNamespace(load_state_dict=self._load_model))
        self.assertEqual(self._run_load(), 3)
        self.assertEqual([event for event in self.events if event[0] == 'model'], [('model', 7), ('model', 8)])
        self.cuda.empty_cache.assert_called_once_with()

    def test_adapter_arguments_are_forwarded(self):
        self.assertEqual(self._run_load(load_arg='mcore_adapter', adapter_name='reference'), 3)
        self.assertEqual(self.filtered_args, [(True, 'reference')])
        self.assertEqual(self.events[0], ('load', 'adapter/iter_0000003'))

    def test_load_failure_preserves_exception(self):
        self.namespace['dist_checkpointing'].load = Mock(side_effect=RuntimeError('checkpoint error'))
        with self.assertRaisesRegex(RuntimeError, 'checkpoint error'):
            self._run_load()
        self.cuda.empty_cache.assert_not_called()
        self.optimizer.load_state_dict.assert_not_called()

    def test_cleanup_inside_loader_fails_lifetime_control(self):
        self.args.no_load_optim = self.args.no_load_rng = True
        self.optimizer.reload_model_params.side_effect = self.real_gc_collect
        with self.assertRaisesRegex(AssertionError, 'checkpoint temporaries still alive'):
            self._run_load()


if __name__ == '__main__':
    unittest.main()
