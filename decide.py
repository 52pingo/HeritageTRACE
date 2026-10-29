"""Portable question-configured Laya inference; validation never loads the model."""
from pathlib import Path
import argparse,json
def validate(questions):
 if not isinstance(questions,dict) or not questions:raise ValueError('Nonempty question mapping required.')
 for name,q in questions.items():
  if q.get('type')!='choice' or not isinstance(q.get('instructions'),str):raise ValueError('Each question needs choice type and written instructions.')
  options=q.get('criteria')
  if not isinstance(options,dict) or not 2<=len(options)<=7:raise ValueError('Use two to seven named criteria.')
  if any(not isinstance(x,str) or not x.strip() for x in options.values()):raise ValueError('Every criterion needs a description.')
 return questions
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--questions',required=True);ap.add_argument('--validate-only',action='store_true');ap.add_argument('--input');ap.add_argument('--output');ap.add_argument('--model');ap.add_argument('--subfolder',default='multilingual');ap.add_argument('--batch-size',type=int,default=8);args=ap.parse_args()
 q=validate(json.loads(Path(args.questions).read_text(encoding='utf8')))
 if args.validate_only:print('Validated',len(q),'questions; no model loaded and no inference performed.');return
 if not all([args.input,args.output,args.model]):ap.error('Inference requires --input --output --model.')
 if args.batch_size<1:ap.error('Batch size must be positive.')
 out=Path(args.output)
 if out.exists():raise FileExistsError('Output exists; use a new versioned path.')
 import laya
 model=laya.load(args.model,subfolder=args.subfolder)
 rows=[json.loads(s) for s in Path(args.input).read_text(encoding='utf8').splitlines() if s.strip()]
 with out.open('x',encoding='utf8') as f:
  for start in range(0,len(rows),args.batch_size):
   batch=rows[start:start+args.batch_size];states=[{'title':r.get('title',''),'summary':r.get('summary','')} for r in batch]
   answers=model.predict_batch(states,q,batch_size=args.batch_size)
   if len(answers)!=len(batch):raise ValueError('Prediction/input count mismatch.')
   for offset,(r,a) in enumerate(zip(batch,answers)):
    aa=a['answers'];result=dict(r);result.update(source_index=start+offset,prob={k:aa[k]['probabilities'] for k in q},pred={k:aa[k]['choice'] for k in q})
    f.write(json.dumps(result,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
