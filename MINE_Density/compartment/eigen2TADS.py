import os
import sys
import argparse

def main(args):
    eigen_folder_path = args.eigen_folder_path
    outdir_path = args.output_folder
    cell_line = args.cell_line
    resolution = args.resolution

    results = dict()
    for chr in range(1, 23):
        eigen_file_path = os.path.join(eigen_folder_path, 'chr{}.eigen.txt'.format(chr))
        with open(eigen_file_path, "r") as f:
            datas = f.readlines()
            for index in range(1, len(datas)):
                if datas[index][0]==datas[index-1][0]:
                    continue
                if chr not in results.keys():
                    results[chr] = list()
                    results[chr].append(0)
                results[chr].append(int(index*resolution))

    for key in results.keys():
        with open(os.path.join(outdir_path, '{}_chr{}'.format(cell_line, key)), "w") as f:
            _list = results[key]
            _list.sort()
            for i in range(len(_list) - 1):
                f.write('chr{}\t{}\t{}\n'.format(key, _list[i], _list[i+1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='用于将juicer eigen的输出转换为SDOC的输入')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='eigen_folder_path', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)
    req_args.add_argument('-c', dest='cell_line', help='', required=True)
    req_args.add_argument('-r', dest='resolution', type=int, help='resolution(b)[default:10000]', default=10000)

    args = parser.parse_args(sys.argv[1:])
    main(args)