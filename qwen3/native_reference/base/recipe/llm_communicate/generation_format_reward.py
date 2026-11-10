"""Opt-in training reward: reject clipped/non-final answers, then score only the final answer."""
import re


def final_answer_fragment(text,answer_format='boxed'):
    if answer_format=='boxed':
        lines=[line.strip() for line in text.splitlines() if line.strip()]
        if not lines:return None
        line=lines[-1]
        # Display-math delimiters may occupy their own lines. They are
        # formatting around the final answer, never trailing answer prose.
        closing={'$$':'$$',r'\]':r'\[',r'\)':r'\('}
        if line in closing:
            opening=next((i for i in range(len(lines)-2,-1,-1) if lines[i]==closing[line]),None)
            if opening is None:return None
            line='\n'.join(lines[opening+1:-1]).strip()
        for left,right in [('$$','$$'),(r'\(',r'\)'),(r'\[',r'\]'),('$','$')]:
            if line.startswith(left) and line.endswith(right) and len(line)>len(left)+len(right):
                line=line[len(left):-len(right)].strip();break
        if not line.startswith(r'\boxed{'):return None
        depth=1;start=len(r'\boxed{')
        for i in range(start,len(line)):
            if i and line[i-1]=='\\':continue
            if line[i]=='{':depth+=1
            elif line[i]=='}':
                depth-=1
                if depth==0:return line if i==len(line)-1 and line[start:i].strip() else None
        return None
    if answer_format=='xml':
        matches=list(re.finditer(r'<answer>\s*([^<>]+?)\s*</answer>',text))
        if not matches or text[matches[-1].end():].strip():return None
        value=matches[-1].group(1).strip()
        return '<answer>'+value+'</answer>' if re.fullmatch('[A-Z]',value) else None
    raise ValueError('Unsupported final answer format: '+answer_format)


def score_with_format(row,*,base_scorer,answer_format='boxed'):
    base=base_scorer(row)
    if base['gold_missing'] or base['scoring_timed_out']:return base
    fragment=final_answer_fragment(row['response'],answer_format)
    ok=row['finish_reason']!='length' and fragment is not None
    strict=base_scorer(dict(row,response=fragment)) if ok else None
    if strict and strict['scoring_timed_out']:return dict(base,scoring_timed_out=True)
    value=strict['acc'] if ok else 0
    return dict(base,acc_unpenalized=base['acc'],score_unpenalized=base['score'],format_ok=ok,
        reward_answer_format=answer_format,acc=value,score=float(value),
        pred=strict.get('pred') if ok else base.get('pred'))
