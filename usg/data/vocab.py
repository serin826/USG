# usg/data/vocab.py
# PSG Dataset vocabulary - 133 object classes (80 thing + 53 stuff), 56 predicates

THING_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

STUFF_CLASSES = [
    'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
    'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net',
    'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand',
    'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged',
    'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged',
    'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
    'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged'
]

# Combined object classes (thing + stuff)
PSG_OBJECTS = THING_CLASSES + STUFF_CLASSES  # 133 classes

PSG_PREDICATES = [
    'over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of',
    'falling off', 'going down', 'painted on', 'walking on', 'running on', 'crossing',
    'standing on', 'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from',
    'wearing', 'holding', 'carrying', 'looking at', 'guiding', 'kissing', 'eating', 'drinking',
    'feeding', 'biting', 'catching', 'picking', 'playing with', 'chasing', 'climbing',
    'cleaning', 'playing', 'touching', 'pushing', 'pulling', 'opening', 'cooking', 'talking to',
    'throwing', 'slicing', 'driving', 'riding', 'parked on', 'driving on', 'about to hit',
    'kicking', 'swinging', 'entering', 'exiting', 'enclosing', 'leaning on'
]  # 56 classes

NUM_OBJECT_CLASSES = len(PSG_OBJECTS)  # 133
NUM_PREDICATE_CLASSES = len(PSG_PREDICATES)  # 56
