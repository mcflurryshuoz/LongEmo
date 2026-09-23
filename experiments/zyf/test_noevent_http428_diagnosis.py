import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.zyf import noevent_http428_diagnosis as driver


class Http428DiagnosisTests(unittest.TestCase):
    def response(self,value,**changes):
        body=json.dumps(value)
        return {'http_status':428,'body_text':body,'body_redacted':False,
                'body_sha256':hashlib.sha256(body.encode()).hexdigest(),**changes}

    def test_dlp_requires_actual_exact_response_evidence(self):
        self.assertEqual(driver.classify({'status':'http_error'},self.response(
            {'error':{'code':428,'message':'Data leak protection rejected'}})),'explicit_dlp_rejection')

    def test_unknown_428_not_generalized(self):
        self.assertEqual(driver.classify({'status':'http_error'},self.response({'error':{'code':428}})),
                         'http428_unclassified')

    def test_redacted_or_changed_body_not_used_for_classification(self):
        r=self.response({'error':{'message':'Data leak protection rejected'}},body_redacted=True)
        self.assertEqual(driver.classify({'status':'http_error'},r),'http_error')
        r['body_redacted']=False;r['body_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'hash changed'):driver.classify({'status':'http_error'},r)

    def test_success_is_not_an_evaluation_result(self):
        self.assertEqual(driver.classify({'status':'validated','reusable':True},None),
                         'validated_payload_not_yet_imported')

    def test_existing_dispatch_prevents_another_even_without_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();d=root/'case';d.mkdir();(d/'request_started.json').write_text('{"request_hash":"same"}')
            with self.assertRaisesRegex(ValueError,'already has'):
                driver.no_prior_dispatch(root,{'failed':{'request_hash':'same'}})
            driver.no_prior_dispatch(root,{'failed':{'request_hash':'other'}})

    def test_changed_first_score_blocks_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp).resolve()/'score.json';p.write_text('original')
            expected={str(p):driver.base.sha(p)};driver.verify_files(expected);p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'first score changed'):driver.verify_files(expected)

    def test_known_denials_excluded_from_fixed_cases(self):
        for n in (9,34,91,99,100):self.assertNotIn(f'G2_V{n:06d}',driver.CASES)
        self.assertEqual(len(driver.CASES),23)


if __name__=='__main__':unittest.main()
