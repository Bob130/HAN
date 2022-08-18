#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json


def generate_json_file(args, json_filename):
    with open(args.data_dir + json_filename, 'w') as f_json:

        file_name = 'informations_troncage_sequences.txt'

        with open(args.data_dir + file_name, 'r') as f_txt:
            label_data = f_txt.read().split('\n')
            label_data.pop()

            data = []
            for line in label_data:
                # id_gesture     id_finger    id_subject    id_essai    beginning_frame     end_frame
                dict_t = {}

                line = line.split()
                line = list(map(int, line))

                id_gesture = line[0]
                id_finger = line[1]
                id_subject = line[2]
                id_essai = line[3]
                beginning_frame = line[4]
                end_frame = line[5]

                skeleton_file = args.data_dir + 'gesture_' + str(id_gesture) + '/' + 'finger_' + str(id_finger) + '/'\
                                + 'subject_' + str(id_subject) + '/' + 'essai_' + str(id_essai) + '/skeleton_world.txt'

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
                dict_t['beginning_frame'] = beginning_frame
                dict_t['end_frame'] = end_frame
                dict_t['skeletons'] = skeletons_list

                dict_t['labels_14'] = id_gesture
                dict_t['labels_28'] = id_gesture * 2 - 1 if id_finger == 1 else id_gesture * 2

                data.append(dict_t)

        json.dump(data, f_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/data/liujianbo/GestureRecognition/DHG2016/', type=str,
                        dest='data_dir', help='The directory of hand pose data.')
    # parser.add_argument('--labeled_data_dir', default='data/LabeledPoseData/', type=str,
    #                     dest='labeled_data_dir', help='The directory of labeled pose data.')

    args = parser.parse_args()

    generate_json_file(args, 'Data.json')



