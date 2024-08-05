import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'




def load_image(imfile):
    img = np.array(cv2.imread(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)



def viz(img, flo, filenames, args):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    flo = flow_viz.flow_to_image(flo)
    print(f'{args.savepath}/{filenames[-7:]}')
    cv2.imwrite(f'{args.savepath}/{filenames[-7:]}', flo[:, :, [2, 1, 0]])


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        t = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            # image0 = cv2.imread(imfile1)

            image1 = load_image(imfile1)
            # print(image1.shape)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            print(image1.shape)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            save_flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
            save_flow_low = flow_low[0].permute(1, 2, 0).cpu().numpy()
            if imfile1[-8] == '\\':
                file_num = imfile1[-7:-4]
            else:
                file_num = imfile1[-8:-4]
            print('imfile1:',imfile1)
            print('save_flow_low',save_flow_low.max())
            print('save_flow_up',save_flow_up.max())
            np.save(args.savepath+'/'+file_num+'.npy',save_flow_up)

            # print(flow_up.shape)
            # break
            viz(image1, flow_up, imfile1, args)

def demo_CW4VS(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        image1 = load_image(args.imagepath1)
        shape = image1.shape
        image2 = load_image(args.imagepath2)


        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)


        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        save_flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
        np.save(args.savepath,save_flow_up[:shape[2], :shape[3]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='./models/raft-sintel.pth')
    parser.add_argument('--source_path', help="dataset rootpath",
                        default="rootpath/dtaset")
    parser.add_argument('--win_size', help="time step",type=int,
                        default=1)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()


    datapath = args.source_path

    imagenames = os.listdir(os.path.join(args.source_path, 'cam%02d'%(1),'images'))

    camNum = len(glob.glob(os.path.join(args.source_path, "cam*")))
    for i in range(camNum):
        t= 0
        for imagename1 in imagenames :
            num,_ = imagename1.split('.')
            imagepath = os.path.join(args.source_path, 'cam%02d' % (i))
            savepath = os.path.join(args.source_path, 'cam%02d'%(i),'flow')
            os.makedirs(savepath,exist_ok=True)
            args.savepath = os.path.join(savepath) + f'/{t}.npy'
            if t <args.win_size:
                last_imagenaem = f"{0}.png"
                imagename = f"{t}.png"
                args.imagepath1 = os.path.join(imagepath,'images',last_imagenaem)
                args.imagepath2 = os.path.join(imagepath,'images',imagename)
            else:
                last_imagenaem = f"{t-args.win_size}.png"
                imagename = f"{t}.png"
                args.imagepath1 = os.path.join(imagepath,'images',last_imagenaem)
                args.imagepath2 = os.path.join(imagepath,'images',imagename)

            demo_CW4VS(args)

            t += 1
