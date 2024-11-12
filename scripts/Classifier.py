import numpy as np

def classify_label(boxes):
    persons = [box for box in boxes if box[5] == 2]
    helmets = [box for box in boxes if box[5] == 3]
    vests = [box for box in boxes if box[5] == 4]
    return persons, helmets, vests

def compute_iou(box1, box2):
    max_top_left_x = max(box1[0], box2[0])
    max_top_left_y = max(box1[1], box2[1])
    min_bottom_right_x = min(box1[2], box2[2])
    min_bottom_right_y = min(box1[3], box2[3])
    max_bottom_right_x = min(box1[2], box2[2])
    max_bottom_right_y = min(box1[3], box2[3])

    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate intersection area
    intersection_width = min_bottom_right_x - max_top_left_x
    intersection_height = min_bottom_right_y - max_top_left_y
    if intersection_width <= 0 or intersection_height <= 0:
        return 0
    intersection_area = intersection_width * intersection_height

    # Calculate union area
    box1_area = (min_bottom_right_x-box1[0]) * (min_bottom_right_y-box1[1])
    box2_area = (max_bottom_right_x-box2[0]) * (max_bottom_right_y-box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def check_compliance(iou_threshold, boxes):
    iou_threshold = 0.5
    compliant_persons = []
    persons, helmets, vests = classify_label(boxes)
    
    for person in persons:
        has_helmet = any(compute_iou(person, helmet) > iou_threshold for helmet in helmets)
        has_vest = any(compute_iou(person, vest) > iou_threshold for vest in vests)
        if not has_helmet:
            print("Person without helmet detected")
        if not has_vest:
            print("Person without vest detected")
        if has_helmet and has_vest:
            compliant_persons.append(person)

    print("Compliant Persons:", compliant_persons)