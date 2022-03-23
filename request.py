#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:34:09 2021

@author: celinejin
"""

import requests
import json

## To test containerized web-service
# url = 'http://127.0.0.1:80/surprise_planner'
## To test local web-service 
#url = 'http://0.0.0.0:80/surprise_planner'
url = 'http://ec2-3-86-91-45.compute-1.amazonaws.com'

# User's inputs
#json_file = open('2019-11-25-135707.json','r+')
#datain = json.load(json_file)
#print(datain)
#r = requests.post(url,json=datain).json()
#dataout = r.json()
#print(r) 
#print(r.json())
#if len(r.json()['note'])>2:
#    print('There is an error happening. Please stop printing!')
file_path = "/Users/celinejin/AFRL/2019-11-25-135707.json"
with open(file_path, 'r') as j:
     datain = json.loads(j.read())
#print(datain)
r = requests.post(url,json=datain)
print(r._content)
