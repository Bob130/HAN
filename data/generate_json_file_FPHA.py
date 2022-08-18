#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json


def generate_json_file(args, json_filename, is_train=True):
    with open(args.data_dir + json_filename, 'w') as f_json:
        if is_train:
            file_name = 'train.txt'
        else:
            file_name = 'test.txt'

        with open(args.data_dir + file_name, 'r') as f_txt:
            label_data = f_txt.read().split('\n')

            data = []
            for line in label_data:
                # data_dir  label
                dict_t = {}

                line = line.split()
                data_dir = line[0]
                label = int(line[1])

                skeleton_file = args.data_dir + 'Hand_pose_annotation_v1/' + data_dir + '/skeleton.txt'

                skeletons_list = []
                with open(skeleton_file, 'r') as f_skeleton:
                    skeletons_data = f_skeleton.read().split('\n')
                    skeletons_data.pop()
                    frames = len(skeletons_data)
                    for skeleton_i in skeletons_data:
                        skeleton = skeleton_i.split()
                        skeleton.pop(0)
                        skeleton = list(map(float, skeleton))
                        skeletons_list.append(skeleton)

                dict_t['data_dir'] = data_dir
                dict_t['label'] = label
                dict_t['size_sequence'] = frames
                dict_t['skeletons'] = skeletons_list

                data.append(dict_t)

        json.dump(data, f_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/data/liujianbo/GestureRecognition/First-Person_Hand_Action/', type=str,
                        dest='data_dir', help='The directory of hand pose data.')
    # parser.add_argument('--labeled_data_dir', default='data/LabeledPoseData/', type=str,
    #                     dest='labeled_data_dir', help='The directory of labeled pose data.')

    args = parser.parse_args()

    generate_json_file(args, 'Train.json', is_train=True)
    generate_json_file(args, 'Test.json', is_train=False)



