import hashlib,json,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.zyf.stage_data import receiver,scp_once,sender,sender_batched

class DataAdmissionTest(unittest.TestCase):
    def test_batch_sends_media_before_marker_on_one_connection(self):
        args,name=self.run_case();(args.videos/name).write_bytes(b'valid video')
        second='G2_V000017.mp4';(args.videos/second).write_bytes(b'another valid video')
        manifest=json.loads(args.manifest.read_text())
        manifest['files'].append({'path':'episode/videos/'+second,'bytes':19,
                                 'sha256':hashlib.sha256(b'another valid video').hexdigest()})
        args.manifest.write_text(json.dumps(manifest))
        args.already_staged=[];args.port=12345;args.known_hosts=args.videos/'known_hosts'
        args.remote_uploads='/task/uploads';args.batch_size=32
        def transfer(command,*_):
            for position,video,payload in [(-5,name,b'valid video'),(-3,second,b'another valid video')]:
                media=Path(command[position]);marker=Path(command[position+1])
                self.assertEqual(media.read_bytes(),payload)
                self.assertEqual(json.loads(marker.read_text()),{'video':video,'upload':media.name})
            return 0
        with patch('experiments.zyf.stage_data.scp_once',side_effect=transfer) as send:
            sender_batched(args)
        self.assertEqual(send.call_count,1)
        self.assertIn(name,json.loads(args.status.read_text())['sent'])
        self.assertIn(second,json.loads(args.status.read_text())['sent'])
        self.assertEqual(list(args.status.parent.glob('upload_batch_*')),[])

    def test_failed_marker_retries_with_unique_names_and_atomic_state(self):
        args,name=self.run_case()
        (args.videos/name).write_bytes(b'valid video')
        args.already_staged=[];args.port=12345;args.known_hosts=args.videos/'known_hosts'
        args.remote_uploads='/task/uploads'
        with patch('experiments.zyf.stage_data.scp_once',side_effect=[0,1,0,0]) as send, \
             patch('experiments.zyf.stage_data.time.sleep'):
            sender(args)
        calls=[c.args[0] for c in send.call_args_list]
        self.assertNotEqual(calls[0][-1],calls[2][-1])
        self.assertNotEqual(calls[1][-1],calls[3][-1])
        self.assertIn(name,json.loads(args.status.read_text())['sent'])
        self.assertFalse(args.status.with_suffix('.tmp').exists())

    def test_interactive_transport_submits_only_one_empty_password(self):
        with tempfile.TemporaryDirectory() as d:
            command=[sys.executable,'-c',"import sys; print('password:',end='',flush=True); assert sys.stdin.readline()=='\\n'"]
            self.assertEqual(scp_once(command,Path(d)/'log',5),0)

    def test_transport_authentication_rejection_is_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            command=[sys.executable,'-c',"import sys,time; print('password:',end='',flush=True); sys.stdin.readline(); print('Permission denied. password:',flush=True); time.sleep(10)"]
            with self.assertRaisesRegex(RuntimeError,'authentication rejected'):
                scp_once(command,Path(d)/'log',5)

    def run_case(self,payload=b'valid video',existing=None):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        uploads=root/'uploads';uploads.mkdir();videos=root/'videos';videos.mkdir()
        content=b'valid video';name='G2_V000016.mp4';upload='a'*32+'.part'
        manifest=root/'manifest.json';manifest.write_text(json.dumps({'files':[{'path':'episode/videos/'+name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}]}))
        (uploads/upload).write_bytes(payload)
        (uploads/('a'*32+'.ready.json')).write_text(json.dumps({'video':name,'upload':upload}))
        if existing is not None:(videos/name).write_bytes(existing)
        args=SimpleNamespace(manifest=manifest,videos=videos,uploads=uploads,status=root/'status.json')
        return args,name

    def test_verified_upload_admitted_atomically(self):
        args,name=self.run_case()
        receiver(args)
        self.assertEqual((args.videos/name).read_bytes(),b'valid video')
        self.assertTrue(json.loads(args.status.read_text())['complete'])
        self.assertEqual(list(args.uploads.iterdir()),[])

    def test_corrupted_upload_never_admitted(self):
        args,name=self.run_case(payload=b'wrong video')
        with patch('experiments.zyf.stage_data.time.sleep',side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):receiver(args)
        self.assertFalse((args.videos/name).exists())
        self.assertEqual(json.loads(args.status.read_text())['verified'],0)
        self.assertEqual(len(list(args.uploads.glob('*.failed'))),1)

    def test_unexpected_existing_video_never_overwritten(self):
        args,name=self.run_case(existing=b'other content')
        with patch('experiments.zyf.stage_data.time.sleep',side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):receiver(args)
        self.assertEqual((args.videos/name).read_bytes(),b'other content')

if __name__=='__main__':unittest.main()
