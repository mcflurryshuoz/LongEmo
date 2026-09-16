import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from unittest.mock import patch

from methods.longemo.embeddings import EventIndex, APIEncoder
from methods.longemo.retrieval import retrieve
from methods.longemo.tests.test_memory import commit
from methods.longemo.memory import empty_memory
from methods.longemo.audio import bridge_audio
from methods.longemo.runner import parser, client_for

class HybridTests(unittest.TestCase):
    def fixture(self):
        memory=commit(empty_memory('video',60,'hash'))
        event=copy.deepcopy(memory['events'][0]);event['id']='E2';event['spans']=[[30,35]];event['summary']='silently worried about illness'
        event['states'][0]['emotion']='anxiety'; event['states'][0]['target']='health'; memory['events'].append(event)
        return memory

    def test_two_routes_rrf_and_no_silent_fallback(self):
        memory=self.fixture();plan={'mode':'local','entity_terms':[],'target_terms':[],'query_terms':[],'time_range':None}
        trace={};result=retrieve(memory,'smiles hearing good news relief',plan,top_k=1,dense_scores={'E1':.1,'E2':.9},trace=trace)
        self.assertEqual(trace['semantic'][0]['id'],'E1');self.assertEqual(trace['dense'][0]['id'],'E2')
        self.assertEqual({x['id'] for x in trace['fused']},{'E1','E2'})
        self.assertEqual({x['id'] for x in result['events']},{'E1','E2'})
        with self.assertRaises(ValueError):retrieve(memory,'question',plan)
        with self.assertRaises(ValueError):retrieve(memory,'question',plan,dense_scores={'E1':.1})

    def test_embedding_chunks_cache_and_revision_invalidation(self):
        class Encoder:
            config={'dimension':2}
            def __init__(self):self.calls=0
            def chunks(self,text):return [text[:30],text[30:]]
            def encode(self,texts,query=False):
                self.calls+=1
                return np.array([[1.,0.]]*len(texts),dtype='float32')
        memory=self.fixture();encoder=Encoder()
        with tempfile.TemporaryDirectory() as tmp:
            a=EventIndex(memory,encoder,tmp);self.assertEqual(a.metadata['chunks'],4)
            b=EventIndex(memory,encoder,tmp);self.assertTrue(b.metadata['cache_reused']);self.assertEqual(encoder.calls,1)
            memory['events'][0]['states'][0]['emotion']='a revised feeling'
            c=EventIndex(memory,encoder,tmp);self.assertNotEqual(a.metadata['signature'],c.metadata['signature'])
            self.assertEqual(set(c.rank('q')),{'E1','E2'})

    def test_api_embedding_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder=APIEncoder('fake',tmp,dimension=3)
            with self.assertRaises(ValueError):encoder._validate([[float('nan'),0,1]],1)
            with self.assertRaises(ValueError):encoder._validate([[0,0,0]],1)
            self.assertTrue(all(len(x)<=6000 for x in encoder.chunks('A'*13000)))

    def test_audio_bridge_never_forwards_raw_audio_to_reasoner(self):
        class Client:
            model='google/audio'
            def configuration(self):return {'model':self.model}
            def generate(self,messages):return {'content':json.dumps({'observations':[{'span':[10,11],'voice':'low voice','cue':'says hello softly','uncertainty':''}]}),'usage':{'total_tokens':10}}
        content=[{'type':'text','text':'frames'},{'type':'input_audio','input_audio':{'data':'ZmFrZQ==','format':'wav'}}]
        with tempfile.TemporaryDirectory() as tmp:
            result,meta=bridge_audio(content,[10,12],Client(),Path(tmp)/'audio.json',1)
            self.assertFalse(any(x['type']=='input_audio' for x in result))
            self.assertIn('hello',json.dumps(result));self.assertIn('source_sha256',meta)

    def test_gpt6_payload_omits_sampling_parameters(self):
        args=parser().parse_args(['build','--data-path','unused','--videos-dir','unused','--output-dir','unused',
            '--model','openai/gpt-6-astra','--base-url','https://openrouter.ai/api/v1'])
        client=client_for(args);payload=client.payload([{'role':'user','content':'hello'}])
        self.assertNotIn('temperature',payload);self.assertEqual(payload['reasoning']['effort'],'medium')
        self.assertEqual(payload['max_completion_tokens'],args.max_tokens)

if __name__=='__main__':unittest.main()
