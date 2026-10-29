"""Weighted lexicon coverage and mapped thematic allocation for scored records."""
from pathlib import Path
import argparse,json
def measure(rows,rules,threshold=.5):
 kept=[r for r in rows if r['prob']['tier']['core']+r['prob']['tier']['peripheral']>=threshold]
 mass=sum(1-r['prob']['tier']['noise'] for r in kept)
 lexical={};mapped={name:0. for name in rules['lexicon']}
 for name,terms in rules['lexicon'].items():
  lexical[name]={}
  for remove in [False,True]:
   terms2=[t for t in terms if not remove or t not in rules['query_terms_removed']]
   numerator=sum(1-r['prob']['tier']['noise'] for r in kept if any(t in (r.get('title','')+' '+r.get('summary','')) for t in terms2))
   lexical[name]['query_removed' if remove else 'full']=100*numerator/mass if mass else None
 for r in kept:
  for label,p in r['prob']['dimension'].items():
   name=rules['dimension_mapping'].get(label)
   if name in mapped:mapped[name]+=(1-r['prob']['tier']['noise'])*p
 total=sum(mapped.values())
 for name in lexical:lexical[name]['model_share']=100*mapped[name]/total if total else None
 return {'records':len(rows),'retained':len(kept),'weighted_record_mass':mass,'threshold':threshold,'values':lexical}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--output',required=True);ap.add_argument('--threshold',type=float,default=.5);a=ap.parse_args()
 if not 0<=a.threshold<=1:ap.error('Threshold must be between zero and one.')
 rows=[json.loads(s) for s in Path(a.input).read_text(encoding='utf8').splitlines() if s.strip()]
 rules=json.loads(Path(a.rules).read_text(encoding='utf8'));result=measure(rows,rules,a.threshold)
 with Path(a.output).open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
if __name__=='__main__':main()
