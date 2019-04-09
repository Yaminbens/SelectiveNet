import json
import os
def open_result(filename):
    with open("results/0406/{}".format(filename), 'r') as fp:
        # json.dump(dict, fp)
        return json.load(fp)


direc = 'results/0406'
files = os.listdir(direc)
for file in files:
    result = open_result(file)
    print(file)
    print(result)
    print()
