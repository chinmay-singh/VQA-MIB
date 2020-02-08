import json

# split = 'val2014'
# split = 'train2014'
split = 'test2015'

match = 27
keyname = 'questions'

with open('v2_OpenEnded_mscoco_' + split + '_questions.json') as f:

    data = json.load(f)

hakku = data[keyname]
for idx, ann in enumerate(hakku):
    print("\ritem: %d" % idx, end='')
    id = ann['image_id']
    if (id == match):
        pakku = ann
        print(ann)
        break

data[keyname] = [pakku]

with open('v2_OpenEnded_mscoco_' + split + '_questions_toy.json', 'w') as fp:
    json.dump(data, fp)

