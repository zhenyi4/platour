import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from qwen3 import grpo_data as data
from qwen3 import native_grpo_validation as native


class PublishedGRPOSplitsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkout = self.root/'checkout'
        table = dict(repositories={'qwen3': dict(repo_id=data.REPO, revision='a'*40)},
                     files={}, protocol_files={})
        for i in range(13,19):
            task = ('chemistry_cot','gsm8k','math500')[(i-13)%3]
            count = 7 if task == 'chemistry_cot' else 503
            folder = Path(f'qwen3/{i:02d}_{task}')
            target = self.checkout/'data'/folder
            target.mkdir(parents=True)
            train = [dict(index=j,prompt_token_ids=[1,j+10,2],gold='A') for j in range(count)]
            test = [dict(index=j,prompt_token_ids=[1,j+9000,2],gold='A') for j in range(2)]
            manifest = dict(task=task,counts={},sha256={})
            for part, rows in [('train',train),('validation',test),('test',test)]:
                path = target/(part+'.jsonl')
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                name = str(folder/path.name)
                table['files'][name] = dict(data.identity(path), group='qwen3',filename=name)
                manifest['counts'][part] = len(rows)
                manifest['sha256'][part] = data.identity(path)['sha256']
            data.write_json(target/'manifest.json', manifest)
            table['protocol_files'][str(folder/'manifest.json')] = data.identity(target/'manifest.json')
        data.write_json(self.checkout/'data/hf_manifest.json', table)
        self.before = copy.deepcopy(table)
        self.bundle = self.root/'bundle'
        self.plan = data.prepare(self.bundle, self.checkout)

    def test_materialized_sizes_partition_and_unchanged_test(self):
        for job in self.plan['jobs'].values():
            want = 2 if job['task']=='chemistry_cot' else 500
            self.assertEqual(job['counts']['validation'], want)
            self.assertEqual(job['counts']['test'],2)
            folder=Path(job['folder'])
            self.assertEqual((self.bundle/'source'/folder/'test.jsonl').read_bytes(),
                             (self.checkout/'data'/folder/'test.jsonl').read_bytes())
            self.assertNotIn(str(folder/'test.jsonl'),self.plan['files'])
            train=data.read_rows(self.bundle/'upload'/folder/'train.jsonl')
            val=data.read_rows(self.bundle/'upload'/folder/'validation.jsonl')
            self.assertEqual(len(set(r['original_train_index'] for r in train+val)),len(train)+len(val))
        self.assertEqual(json.loads((self.checkout/'data/hf_manifest.json').read_text()),self.before)
        self.assertEqual(len(self.plan['files']),25)
        data.validate(self.bundle)

    def test_tampered_data_and_repeat_partition_are_rejected(self):
        job=self.plan['jobs']['16']; folder=Path(job['folder'])
        train=data.read_rows(self.bundle/'upload'/folder/'train.jsonl')
        test=data.read_rows(self.bundle/'source'/folder/'test.jsonl')
        with self.assertRaisesRegex(ValueError,'already partitioned'):
            data.partition(train,test,'chemistry_cot')
        (self.bundle/'upload'/folder/'validation.jsonl').write_text('[]\n')
        with self.assertRaises(ValueError): data.validate(self.bundle)

    def test_network_failure_leaves_registry_and_local_payloads_unchanged(self):
        with patch('huggingface_hub.HfApi') as cls:
            cls.return_value.repo_info.side_effect=ConnectionError('offline')
            with self.assertRaises(ConnectionError): data.publish(self.bundle,self.checkout)
            cls.return_value.create_commit.assert_not_called()
        self.assertFalse((self.bundle/'hf-receipt.json').exists())
        self.assertEqual(json.loads((self.checkout/'data/hf_manifest.json').read_text()), self.before)

    def test_publish_is_atomic_and_activates_verified_revision_only(self):
        def download(repo,name,**kw):
            self.assertEqual(repo,data.REPO)
            self.assertEqual(kw['repo_type'],'dataset')
            if kw['revision']=='c'*40 and name in self.plan['files']:
                return str(self.bundle/'upload'/name)
            return str(self.checkout/'data'/name)
        with patch('huggingface_hub.HfApi') as cls, \
                patch('huggingface_hub.hf_hub_download',side_effect=download):
            cls.return_value.repo_info.return_value=SimpleNamespace(sha='b'*40)
            cls.return_value.create_commit.return_value=SimpleNamespace(oid='c'*40)
            receipt=data.publish(self.bundle,self.checkout)
            call=cls.return_value.create_commit.call_args.kwargs
            self.assertEqual(call['parent_commit'],'b'*40)
            self.assertEqual({op.path_in_repo for op in call['operations']},set(self.plan['files']))
        self.assertTrue(receipt['remote_verified'])
        current=json.loads((self.checkout/'data/hf_manifest.json').read_text())
        self.assertEqual(current['repositories']['qwen3']['revision'],'c'*40)
        for job in self.plan['jobs'].values():
            folder=Path(job['folder'])
            for part in ('train','validation'):
                self.assertEqual((self.checkout/'data'/folder/(part+'.jsonl')).read_bytes(),
                                 (self.bundle/'upload'/folder/(part+'.jsonl')).read_bytes())
            name=str(folder/'test.jsonl')
            self.assertEqual(current['files'][name],self.before['files'][name])
        data.activate(self.bundle,self.checkout)  # Idempotent after successful publication.

    def test_consumers_keep_published_rows_and_allow_different_val_test_sizes(self):
        from qwen3.runner import prepare_grpo_holdout
        job=self.plan['jobs']['17']; folder=Path(job['folder'])
        train=data.read_rows(self.bundle/'upload'/folder/'train.jsonl')
        val=data.read_rows(self.bundle/'upload'/folder/'validation.jsonl')
        test=data.read_rows(self.bundle/'source'/folder/'test.jsonl')
        config=dict(method='grpo',batch_size=2,eval_samples=2)
        actual_train,actual_val,protocol,split=prepare_grpo_holdout(train,test,config,validation_rows=val)
        self.assertEqual((actual_train,actual_val),(train,val))
        self.assertEqual((split['validation_questions'],split['test_questions']),(500,2))
        with self.assertRaisesRegex(ValueError,'refusing to resplit'):
            prepare_grpo_holdout(train,test,config)
        root=self.root/'native'
        native.save(root/'train_prompts.json',train)
        native.save(root/'eval_prompts.json',test)
        protocol=dict(batch_size=2,epochs=3)
        native.configure(root,protocol,validation_rows=val)
        native.verify_inputs(root,protocol)
        self.assertEqual(json.loads((root/'train_prompts.json').read_text()),train)
        self.assertEqual(json.loads((root/'eval_prompts.json').read_text()),val)
        self.assertEqual((protocol['eval_questions'],protocol['test_questions']),(500,2))

    def test_gsm8k_validation_avg4_and_test_avg2_are_independent(self):
        from qwen3.cli import select
        from qwen3.runner import prepare_grpo_holdout, validation_without_rng_side_effect
        from unittest.mock import patch
        train = [dict(index=0, original_train_index=0, prompt_token_ids=[1, 10, 2], gold='A')]
        val = [dict(index=i, original_train_index=i+1, prompt_token_ids=[1, 100+i, 2], gold='A') for i in range(500)]
        test = [dict(index=i, prompt_token_ids=[1, 10000+i, 2], gold='A') for i in range(1319)]
        for experiment in (14, 17):
            config = select(experiment)
            self.assertEqual((config['validation_samples'], config['eval_samples']), (4, 2))
            _, _, active, split = prepare_grpo_holdout(train, test, config, validation_rows=val)
            self.assertEqual((active['eval_samples'], active['test_eval_samples']), (4, 2))
            self.assertEqual((len(val)*split['validation_samples'],len(test)*split['test_samples']), (2000,2638))
            active['eval_backend']='vllm'
            with patch('qwen3.runner.evaluate_vllm', return_value={'responses': 2000}) as evaluate:
                validation_without_rng_side_effect(None, None, val, active, self.root/'eval')
            self.assertEqual(evaluate.call_args.kwargs['count'], 4)
            root = self.root/str(experiment)
            native.save(root/'train_prompts.json',train)
            native.save(root/'eval_prompts.json',test)
            protocol=dict(batch_size=64,epochs=3,eval_samples=2,eval_seed_stride=2,
                          validation_samples=config['validation_samples'])
            native.configure(root,protocol,validation_rows=val)
            self.assertEqual((protocol['eval_samples'],protocol['eval_seed_stride']), (4,4))
            self.assertEqual((protocol['test_eval_samples'],protocol['test_eval_seed_stride']), (2,2))
            native.save(root/'protocol.json',protocol)
            before=(root/'protocol.json').read_bytes()
            destination=native.test_evaluation_directory(root,protocol)
            test_protocol=json.loads((destination.parent/'protocol.json').read_text())
            self.assertEqual((test_protocol['eval_samples'],test_protocol['eval_seed_stride'],test_protocol['eval_questions']), (2,2,1319))
            self.assertEqual((root/'protocol.json').read_bytes(),before)


if __name__=='__main__': unittest.main()
