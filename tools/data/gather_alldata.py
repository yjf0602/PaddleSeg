import argparse
import json
import os.path
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    return parser.parse_args()


def gather_alldata(args):
    """将分散在多个子文件夹中的数据集中到一个文件夹中"""
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    file_index = 0

    input_dir = args.input_dir
    folders = os.listdir(input_dir)
    for folder in folders:
        folder = input_dir + '/' + folder
        if not os.path.isdir(folder):
            continue
        files = os.listdir(folder)
        img_files = list(filter(lambda x: '.png' in x, files))
        for img_file in img_files:
            img_file_path = folder + '/' + img_file
            json_file_path = img_file_path.replace('.png', '.json')
            if not os.path.exists(json_file_path):
                continue
            index_file_text = str(file_index).zfill(5)
            target_img_file_path = output_dir + '/' + index_file_text + '.png'
            target_json_file_path = output_dir + '/' + index_file_text + '.json'
            print("copy: {} to {}".format(img_file_path, target_img_file_path))
            shutil.copy(img_file_path, target_img_file_path)
            # 修改 json 文件 imagePath 属性
            with open(json_file_path) as json_file:
                json_data = json.load(json_file)
                json_data['imagePath'] = index_file_text + '.png'
                with open(target_json_file_path,"w") as tar_file:
                    json.dump(json_data, tar_file)

            file_index = file_index + 1


if __name__ == '__main__':
    args = parse_args()
    gather_alldata(args)
