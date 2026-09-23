"""Offline tests: delete unsupported context records without inventing evidence."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from experiments.zyf import noevent_core_boundary_continuation as boundary
from experiments.zyf.matched_pilot import read, sha, write, memory_root
from methods.longemo.noevent_memory import apply_window, empty_memory


class CoreBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.memory=empty_memory('V',60,'video-sha')
        self.payload={'entities':[{'id':'p','name':None,'description':'visible person'}],
            'observations':[
                {'id':'in','subject':'p','span':[21,39],'cue':'observable movement','modality':'visual'},
                {'id':'overlap','subject':'p','span':[39,41],'cue':'another movement','modality':'visual'},
                {'id':'outside','subject':'p','span':[40,40],'cue':'at the next boundary','modality':'visual'}],
            'summary':'A person moves.', 'actions':[], 'participants':['p'],
            'emotion_cues':[
                {'subject':'p','target':'other','emotion':'uncertain','intensity':'low','evidence_ids':['in']},
                {'subject':'p','target':'other','emotion':'uncertain','intensity':'low','evidence_ids':['in','outside']}]}
        self.runner=SimpleNamespace(apply_window=apply_window)

    def validate(self,payload=None):
        return boundary.validate_projection(self.runner,self.memory,payload or self.payload,
                                             'W00002',[20,40],[18,42],{'audio':False})

    def test_retained_values_unchanged_and_original_not_mutated(self):
        before=copy.deepcopy(self.payload)
        value,updated,details=self.validate()
        self.assertEqual(self.payload,before)
        self.assertEqual(value['observations'],before['observations'][:2])
        self.assertEqual(value['emotion_cues'],before['emotion_cues'][:1])
        self.assertEqual(details['removed_observation_ids'],['outside'])
        self.assertEqual(len(updated['observations']),2)
        self.assertEqual(self.memory['completed_windows'],[])

    def test_span_starting_at_core_end_is_removed_without_shift(self):
        self.payload['observations'][-1]['span']=[40,42]
        value,_,_=self.validate()
        self.assertEqual([o['span'] for o in value['observations']],[[21,39],[39,41]])

    def test_left_padding_only_is_removed(self):
        self.payload['observations'][-1]['span']=[18,19.9]
        self.assertEqual(self.validate()[2]['removed_observation_ids'],['outside'])

    def test_valid_overlap_at_left_boundary_is_not_changed(self):
        self.payload['observations'][0]['span']=[18,20]
        self.assertEqual(self.validate()[0]['observations'][0]['span'],[18,20])

    def test_valid_window_is_never_reprocessed(self):
        self.payload['observations'][-1]['span']=[39,40]
        with self.assertRaisesRegex(ValueError,'valid responses'):
            self.validate()

    def test_out_of_media_and_nan_are_not_fixed(self):
        for span in ([42,43],[float('nan'),40]):
            p=copy.deepcopy(self.payload);p['observations'][-1]['span']=span
            with self.assertRaises(ValueError):self.validate(p)

    def test_unknown_evidence_is_not_silently_deleted(self):
        self.payload['emotion_cues'][-1]['evidence_ids']=['unknown']
        with self.assertRaisesRegex(ValueError,'unknown emotion evidence'):self.validate()

    def test_empty_core_result_is_not_accepted(self):
        for item in self.payload['observations']:item['span']=[40,40]
        with self.assertRaisesRegex(ValueError,'retain supported'):self.validate()

    def test_duplicate_ids_not_masked(self):
        self.payload['observations'][-1]['id']='in'
        with self.assertRaises(ValueError):self.validate()

    def test_graph_field_or_other_schema_error_remains_invalid(self):
        self.payload['events']=[]
        with self.assertRaisesRegex(ValueError,'not exclusively'):self.validate()

    def test_another_error_after_projection_still_rejected(self):
        self.payload['participants']=['unknown-person']
        with self.assertRaisesRegex(ValueError,'participant'):self.validate()

    def test_seed_writes_once_and_proves_no_api(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp).resolve();folder=memory_root(run,'noevent','V')/'V'
            write(folder/'memory.json',self.memory)
            write(folder/'audio/W00002.json',{'cached':True})
            value,updated,details=self.validate()
            artifact={'window':'W00002','payload':value,'updated':updated,'context':{},'sampling':{},
                'details':details,'request_hash':'final','initial_request_hash':'initial',
                'diagnostic_hashes':{},'parent_memory_sha256':sha(folder/'memory.json'),
                'pending_audio_sha256':sha(folder/'audio/W00002.json')}
            receipt=boundary.seed_window(run,'V',{},artifact)
            self.assertTrue(receipt['no_api_calls'])
            self.assertEqual(read(folder/'windows/W00002.json')['perception'],value)
            self.assertEqual(read(folder/'memory.json')['completed_windows'],['W00002'])
            with self.assertRaises(ValueError):boundary.seed_window(run,'V',{},artifact)


if __name__=='__main__':unittest.main()
