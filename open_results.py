import json

def open_result(filename):
    with open("results\{}".format(filename), 'r') as fp:
        # json.dump(dict, fp)
        return json.load(fp)


result = open_result('full_cov_2.json')
print()