from argparse import Namespace
import numpy as np
import json

def process_args(args:Namespace)-> dict:
    processed_args = {}
    json_content = {}

    
    if args.json:
        with open(args.json,"r") as f:
            json_content = json.load(f)

    processed_args["sdf"] = json_content.get("sdf",args.sdf)
    processed_args["numSteps"] = json_content.get("numSteps",args.numSteps)
    processed_args["triangle"] = np.array(json_content.get("triangle", json.loads(args.triangle) if type(args.triangle) == str else args.triangle))
    processed_args["start"] = np.array(json_content.get("start",args.start))
    processed_args["plot"] = json_content.get("plot",args.plot)
    processed_args["interactive"] = False

    return processed_args
    
