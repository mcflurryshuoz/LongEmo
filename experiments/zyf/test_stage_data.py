import hashlib,json,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from experiments.zyf.stage_data import receiver,scp_once

class DataAdmissionTest(unittest.TestCase):
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
