import os
import sys
import argparse
from math import sqrt, ceil, log10
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


def fftblur(img, sigma):
    h, w = img.shape

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X, Y = X-w//2, Y-h//2
    Z = np.exp(-0.5*(X**2 + Y**2)/(sigma**2))
    Z = Z/Z.sum()

    out = ifftshift(ifft2(fft2(img)*fft2(Z)))
    return out


def fan_func(sparse_img, mask):
    N = np.prod(mask.shape)
    num_kept = np.nonzero(mask)[0].shape[0]
    sigma = sqrt(N/(np.pi*num_kept))

    c = fftblur(sparse_img, sigma)
    i = fftblur(mask, sigma)

    img = np.abs(c/i)
    return img


def sparsify(img, mask, sparse_img, span, col_num):
    mask[:, :] = 1
    dim = img.shape
    h, w = dim[0], dim[1]
    for i in range(0, h, span):
        for j in range(0, w, span):
            tmp = img[i:i+span, j:j+span].mean()
            _x, _y = i + span / 2, j + col_num*w + span / 2
            _length = max(abs(_x - _y), 1)
            tmp *= max(log10(_length), 1)
            sparse_img[i:i+span, j:j+span] += tmp
    return sparse_img, mask


def create_low_matrix(high_matrix, col_num, spans=[5, 10]):
    h, w = high_matrix.shape
    mask = np.zeros((h, w))
    sparse_matrix = np.zeros((h, w))

    spans.sort(reverse=True)
    for span in spans:
        sparse_matrix, mask = sparsify(high_matrix, mask, sparse_matrix, span, col_num)
    replaced_matrix = fan_func(sparse_matrix, mask)

    return replaced_matrix


def main(args):
    hic_dir = args.input_folder
    out_dir = args.output_folder
    subimage_size = args.subimage_size
    focus_size = args.focus_size
    print(args)
    if focus_size % subimage_size != 0:
        raise Exception()

    hr_dir = os.path.join(out_dir, 'hr')
    replaced_dir = os.path.join(out_dir, 'replaced')
    mkdir(out_dir)
    mkdir(hr_dir)
    mkdir(replaced_dir)

    hic_file_path = []
    for root, dirs, files in os.walk(hic_dir):
        if len(files) <= 0:
            continue
        for f in files:
            hic_file_path.append(os.path.join(root, f))
    
    for _p in hic_file_path:
        print(_p)
        prefix, ext = os.path.splitext(os.path.basename(_p))
        hic = np.load(_p)['hic']
        rows, cols = hic.shape
        
        sub_rows, sub_cols = ceil(rows / subimage_size), round(focus_size / subimage_size)
        hr_hic = np.zeros((sub_rows, sub_cols, subimage_size, subimage_size))
        replaced_hic = np.zeros((sub_rows, sub_cols, subimage_size, subimage_size))
        for m in range(sub_rows):
            i = m * subimage_size
            # offset = max(round(m - focus_size / subimage_size + 1), 0)
            # offset = min(sub_rows - sub_cols - 1, offset)
            offset = m
            for n in range(sub_cols):
                j = (offset + n) * subimage_size
                if j >= cols:
                    break
                high_matrix = hic[i:i+subimage_size, j:j+subimage_size]
                # 补零
                _y = i + subimage_size - rows if i + subimage_size > rows else 0
                _x = j + subimage_size - cols if j + subimage_size > cols else 0
                high_matrix = np.pad(high_matrix, ((0,_y),(0,_x)), 'constant', constant_values=(0, 0))

                replaced_matrix = create_low_matrix(high_matrix, j, spans=[5, 10])

                hr_hic[m, n] = high_matrix
                replaced_hic[m, n] = replaced_matrix

        prefix = prefix.lower()
        np.savez_compressed(os.path.join(hr_dir, prefix), hic=hr_hic)
        np.savez_compressed(os.path.join(replaced_dir, prefix), hic=replaced_hic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate hic train data')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-i', dest='input_folder', help='', required=True)
    req_args.add_argument('-o', dest='output_folder', help='', required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-s', dest='subimage_size', type=int,
                           help='The size of the captured image[default:400]',
                           default=400)
    misc_args.add_argument('-f', dest='focus_size', type=int,
                           help='The size of the picture to follow[default:2000]',
                           default=2000)
    
    args = parser.parse_args(sys.argv[1:])
    main(args)