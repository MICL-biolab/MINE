import os
import sys
import argparse

def main(args):
    TAD_file_path = args.TAD_file_path
    outdir_path = args.output_folder
    cell_line = args.cell_line

    results = dict()
    with open(TAD_file_path, "r") as f:
        datas = f.readlines()
        for data in datas:
            _data = data.split()
            if int(_data[5]) != 1:
                continue
            chr, bin1, bin2 = int(_data[0]), int(_data[1]), int(_data[2])
            if chr not in results.keys():
                results[chr] = list()
            results[chr].append(int((bin1+bin2)/2))

    for key in results.keys():
        with open(os.path.join(outdir_path, '{}_chr{}'.format(cell_line, key)), "w") as f:
            _list = results[key]
            _list.sort()
            for i in range(len(_list) - 1):
                f.write('chr{}\t{}\t{}\n'.format(key, _list[i], _list[i+1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='用于将HiCDB的输出转换为SDOC的输入')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='TAD_file_path', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)
    req_args.add_argument('-c', dest='cell_line', help='', required=True)

    args = parser.parse_args(sys.argv[1:])
    main(args)