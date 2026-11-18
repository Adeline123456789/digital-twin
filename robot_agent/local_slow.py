"""Optional loopback-only llama.cpp diagnosis. No tools or generated code execution."""
import http.client
import json
import time
from .supervisor import CHECKS,CATEGORIES


def local_diagnosis(packet,tokens,seconds):
    deadline=time.monotonic()+seconds
    def request(path,data):
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise TimeoutError()
        client=http.client.HTTPConnection('127.0.0.1',8080,timeout=remaining)
        try:
            client.request('POST',path,json.dumps(data),{'Content-Type':'application/json'})
            result=client.getresponse()
            if result.status!=200:
                raise OSError('Local diagnosis unavailable')
            return json.loads(result.read(65536))
        finally:
            client.close()
    prompt=('Suggest a hypothesis and at most three offline checks. Return only JSON with keys category and checks. '
            +json.dumps(dict(categories=sorted(CATEGORIES),checks=CHECKS,capsule=packet)))
    ids=request('/tokenize',dict(content=prompt,add_special=True))['tokens']
    remaining=tokens-len(ids)
    if remaining<=0:
        raise ValueError('Input exceeds remaining token budget')
    result=request('/completion',dict(prompt=ids,n_predict=min(192,remaining),temperature=0,stream=False))
    value=json.loads(result['content'])
    return dict(category=value['category'],checks=value['checks'],input_tokens=len(ids),
                output_tokens=result['tokens_predicted'])
