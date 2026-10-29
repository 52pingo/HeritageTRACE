"""Offline public-release checks; no pretrained weights or platform access."""
import importlib.util,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module(path):
 spec=importlib.util.spec_from_file_location(path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class ReleaseTests(unittest.TestCase):
 def test_domain_contracts(self):
  decide=module(ROOT/'decide.py')
  for p in (ROOT/'domains').glob('*.json'):self.assertTrue(decide.validate(json.loads(p.read_text(encoding='utf8'))))
  with self.assertRaises(ValueError):decide.validate({'bad':{'type':'choice','instructions':'x','criteria':{'only':'one option'}}})
 def test_measurement_weights_and_query_removal(self):
  m=module(ROOT/'analysis/measure.py')
  rules={'lexicon':{'history':['history','query']},'query_terms_removed':['query'],'dimension_mapping':{'history':'history'}}
  rows=[{'title':'query','summary':'','prob':{'tier':{'core':.8,'peripheral':0,'noise':.2},'dimension':{'history':1}}},{'title':'history','summary':'','prob':{'tier':{'core':.6,'peripheral':0,'noise':.4},'dimension':{'history':1}}}]
  out=m.measure(rows,rules);self.assertEqual(out['retained'],2);self.assertAlmostEqual(out['values']['history']['full'],100);self.assertAlmostEqual(out['values']['history']['query_removed'],100*.6/1.4)
  self.assertIsNone(m.measure([],rules)['values']['history']['full'])
 def test_collector_cli_and_offline_pipeline(self):
  p=subprocess.run([sys.executable,str(ROOT/'collector/main.py'),'--platforms','wikipedia','--keywords','heritage','--mode','quick','--mock','--non-interactive','--export','none','--run-id','release-test'],cwd=ROOT,capture_output=True)
  self.assertEqual(p.returncode,0,p.stderr.decode(errors='replace'));self.assertIn(b'L1=12',p.stdout)
 def test_live_adapter_registry_imports_without_network(self):
  code="import sys;sys.path.insert(0,'collector');from config import Settings;from platforms import create_collectors;c=create_collectors(Settings(),['wikipedia','reddit']);assert set(c)=={'wikipedia','reddit'}"
  p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True);self.assertEqual(p.returncode,0,p.stderr.decode(errors='replace'))
if __name__=='__main__':unittest.main()
