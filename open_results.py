import json
import os
def open_result(filename):
    with open("results/{}".format(filename), 'r') as fp:
        # json.dump(dict, fp)
        return json.load(fp)


direc = 'results'
files = os.listdir(direc)
for file in files:
    if file[0:8] == 'coverage':
        result = open_result(file)
        print(file)
        print(result)
        print()
