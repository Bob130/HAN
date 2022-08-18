#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json


def generate_json_file(args, json_filename, is_train=True):
    with open(args.data_dir + json_filename, 'w') as f_json:
        if is_train:
            file_name = 'train_gestures.txt'
        else:
            file_name = 'test_gestures.txt'

        with open(args.data_dir + file_name, 'r') as f_txt:
            label_data = f_txt.read().split('\n')
            label_data.pop()

            data = []
            for line in label_data:
                # id_gesture     id_finger    id_subject    id_essai    14_labels    28_labels    size_sequence
                dict_t = {}

                line = line.split()
                line = list(map(int, line))

                id_gesture = line[0]
                id_finger = line[1]
                id_subject = line[2]
                id_essai = line[3]
                labels_14 = line[4]
                labels_28 = line[5]
                size_sequence = line[6]

                skeleton_file = args.data_dir + 'gesture_' + str(id_gesture) + '/' + 'finger_' + str(id_finger) + '/'\
                                + 'subject_' + str(id_subject) + '/' + 'essai_' + str(id_essai) + '/skeletons_world.txt'

                skeletons_list = []
                with open(skeleton_file, 'r') as f_skeleton:
                    skeletons_data = f_skeleton.read().split('\n')
                    skeletons_data.pop()
                    for skeleton_i in skeletons_data:
                        skeleton = skeleton_i.split()
                        skeleton = list(map(float, skeleton))
                        skeletons_list.append(skeleton)

                dict_t['id_gesture'] = id_gesture
                dict_t['id_finger'] = id_finger
                dict_t['id_subject'] = id_subject
                dict_t['id_essai'] = id_essai
                dict_t['labels_14'] = labels_14
                dict_t['labels_28'] = labels_28
                dict_t['size_sequence'] = size_sequence
                dict_t['skeletons'] = skeletons_list

                data.append(dict_t)

        json.dump(data, f_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/data/liujianbo/GestureRecognition/HandGestureDataset_SHREC2017/', type=str,
                        dest='data_dir', help='The directory of hand pose data.')
    # parser.add_argument('--labeled_data_dir', default='data/LabeledPoseData/', type=str,
    #                     dest='labeled_data_dir', help='The directory of labeled pose data.')

    args = parser.parse_args()

    generate_json_file(args, 'Train.json', is_train=True)
    generate_json_file(args, 'Test.json', is_train=False)



