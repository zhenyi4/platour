"""CPU tests for remote high-level judging without Docker on training nodes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from qwen35_bridge_opd_core import save
from shared_code_judge_client import SharedCodeScorer, service_info, within
from shared_code_judge_service import Service
import train_qwen35_bridge_opd_vllm as trainer


class Tests(unittest.TestCase):
    def test_paths_outside_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):within(Path(tmp).parent/'other',tmp)

    def test_stale_heartbeat_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            save(Path(tmp)/'ready.json',dict(ready=True,backend='shared_files_docker',spec_id='test',heartbeat=time.time()-100))
            with self.assertRaises(RuntimeError):service_info(tmp)

    def test_lcb_rpc_uses_shared_service_without_spawning(self):
        with patch('shared_code_judge_client.SharedCodeScorer',return_value='remote') as shared, \
             patch.object(trainer,'LocalRPC') as local:
            result=trainer.RPC('lcb_score_worker.py','test',boxed_only=False)
        self.assertEqual(result,'remote')
        shared.assert_called_once()
        local.assert_not_called()

    def test_gpu_and_math_rpc_still_local(self):
        with patch.object(trainer,'LocalRPC',return_value='local') as local:
            self.assertEqual(trainer.RPC('qwen35_bridge_opd_vllm_worker.py','test',gpu=4),'local')
            self.assertEqual(trainer.RPC('qwen35_bridge_opd_score_worker.py','test',boxed_only=False),'local')
        self.assertEqual(local.call_count,2)

    def test_service_fixed_official_spec_and_penalty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir()
            save(root/'protocol.json',dict(dataset='lcb',judge_service_spec_id='spec'))
            rows=[dict(index=0,sample_index=i,key='known',judge=dict(kind='lcb',key='tampered',tests=999),
                       response='text',tokens=10,finish_reason='length' if i==1 else 'stop') for i in range(2)]
            save(root/'inputs.json',dict(complete=True,rows=rows))
            ident='a'*32;request=root/f'{ident}.json'
            save(request,dict(id=ident,root=str(root),inputs=[str(root/'inputs.json')],result=str(root/'summary.json'),
                expected_responses=2,format_penalty=True,spec_id='spec',service_instance='instance',deadline=time.time()+100))
            service=Service.__new__(Service)
            service.allowed_roots=Path(tmp);service.spec_id='spec';service.instance='instance';service.problems={'known':{'tests':1}}
            seen=[]
            def judge(row):
                seen.append(row['judge'])
                return dict(row,acc=1,score=1.,no_code=False,judge_error_code=0)
            with ThreadPoolExecutor(max_workers=2) as pool,patch('lcb_score_worker.judge_row',side_effect=judge):
                result=service.request(request,pool)
            self.assertTrue(result['complete'])
            self.assertTrue(all(spec==dict(kind='lcb',key='known',tests=1) for spec in seen))
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['accuracy'],.5)
            self.assertEqual(summary['accuracy_unpenalized'],1.)
            self.assertEqual(summary['judge_service_backend'],'shared_files_docker')
            with ThreadPoolExecutor(max_workers=1) as pool:
                with self.assertRaises(ValueError):service.request(request,pool)

    def test_reply_failure_not_wrong_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp);root=prefix/'run';service=prefix/'service';root.mkdir();service.mkdir()
            info=dict(ready=True,backend='shared_files_docker',spec_id='spec',instance='instance',heartbeat=time.time())
            save(service/'ready.json',info)
            save(root/'protocol.json',dict(dataset='lcb',judge_service_spec_id='spec'))
            save(root/'inputs.json',dict(complete=True,rows=[]))
            ident='b'*32
            save(service/'replies'/f'{ident}.json',dict(id=ident,complete=False,error='sandbox failure'))
            with patch('shared_code_judge_client.ALLOWED_ROOTS',prefix),patch('shared_code_judge_client.uuid.uuid4') as uuid:
                uuid.return_value.hex=ident
                client=SharedCodeScorer(root/'train/score_worker',service)
                with self.assertRaisesRegex(RuntimeError,'sandbox failure'):
                    client.call(dict(inputs=[str(root/'inputs.json')],expected_responses=1,result=str(root/'summary.json')))
            self.assertFalse((root/'summary.json').exists())


if __name__=='__main__':
    unittest.main()
