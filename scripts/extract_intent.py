import json
import os

if __name__=="__main__":
    script_path = os.path.abspath(__file__)
    file_path = os.path.join(os.path.dirname(os.path.dirname(script_path)),'data','conala-corpus','conala-mined.jsonl')
    intents = []
    with open(file_path,'r') as f:
        for line in f:
            intents.append(json.loads(line)['intent'])

    outpath = os.path.join(os.path.dirname(file_path),'conala-mined-intents.txt')
    with open(outpath,'w') as f:
        f.write('\n'.join(intents))
