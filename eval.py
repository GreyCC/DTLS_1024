import torch
import torch.nn.functional as F
import os
import numpy as np
import argparse

from tqdm import tqdm
from torch.utils import data
from models import Encoder, Decoder
from operation import InfiniteSamplerWrapper
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
from pathlib import Path

def downgrade_in_seq(image, target_size, domain_list):
    ori_size = domain_list[0]

    image_a = F.interpolate(image, size=domain_list[target_size], mode='bicubic', antialias=True)
    image_a = F.interpolate(image_a, size=ori_size, mode='bicubic', antialias=True)

    ## Anti Alpha channel in .png format
    if image_a.shape[1] == 4:
        image_a = image_a[:,:3,:,:]
    return image_a

def LR_for_showing(image, domain_list):
    ori_size = domain_list[0]

    for j in range(image.shape[0]):
        image_a = image[j].clone().unsqueeze(0)
        image_a = F.interpolate(image_a, size=domain_list[-1], mode='nearest-exact')
        image[j] = F.interpolate(image_a, size=ori_size, mode='nearest-exact').squeeze()
    return image

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img), str(path)

transform = transforms.Compose([
    transforms.Lambda(lambda t: (t * 2) - 1)
])


def inference(args, domains_interval):
    data_root = args.path
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    input_folder = args.input_folder
    dataloader_workers = args.workers
    saved_image_folder = f"eval/{args.output_path}"
    if not os.path.exists(saved_image_folder):
        os.makedirs(saved_image_folder)

    num_domains = len(domains_interval)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Select a GPU runtime in Colab or use a CUDA-enabled PyTorch installation."
        )
    device = torch.device(f"cuda:{args.cuda}")

    dataset = Dataset(data_root, im_size)
    dataloader = iter(
        DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers, pin_memory=False))

    encoder = Encoder()
    decoder = Decoder()

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint, map_location=device)
        encoder.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['enc'].items()})
        decoder.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['dec'].items()})
        del ckpt

    encoder.to(device)
    decoder.to(device)

    samples = dataset.__len__()

    if input_folder is None:
        for n in tqdm(range(samples)):

            gt_img, paths = next(dataloader)
            gt_img = gt_img.to(device)
            current_batch_size = gt_img.shape[0]

            # Get the exact original image filename (e.g., 'sample_01.jpg')
            # DataLoader returns a list/tuple of paths for the batch
            file_path = paths[0]
            image_name = Path(file_path).name  # Keeps the original filename and extension

            lr_img = downgrade_in_seq(gt_img.clone(), num_domains - 1, domains_interval)

            with torch.no_grad():
                sr_img = lr_img.clone()
                previous_prediction = None
                inverse_momentum = 0
                for i in reversed(range(1, num_domains)):
                    t = torch.ones(current_batch_size, device=device).long() * i
                    if previous_prediction is None:
                        inverse_momentum_LR = 0
                    else:
                        inverse_momentum_LR = downgrade_in_seq(inverse_momentum.clone(), t, domains_interval)

                    l, h = encoder(sr_img + inverse_momentum_LR, t)

                    result = decoder(l, t, h)
                    temp_result = result.clone()
                    if previous_prediction is None:
                        previous_prediction = temp_result.clone()
                    inverse_momentum += (previous_prediction - temp_result)
                    previous_prediction = temp_result.clone()

                    if i == 4:
                        break
                    else:
                        sr_img = downgrade_in_seq(result, t - 1, domains_interval)

            result = F.interpolate(result, size=512, mode='bicubic', antialias=True)

            # Save the result with the exact source image name
            save_path = os.path.join(saved_image_folder, image_name)
            vutils.save_image(result.add(1).mul(0.5), save_path)

    # else:
    #     import glob
    #     import cv2
    #
    #     for idx, path in enumerate(sorted(glob.glob(os.path.join(input_folder, '*')))):
    #         imgname = os.path.splitext(os.path.basename(path))[0]
    #         img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    #         img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    #         # gt_img = transform(img.unsqueeze(0)).to(device)
    #         img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
    #         img = img.unsqueeze(0).to(device)
    #
    #         gt_img = img * 2 - 1
    #         target_domain = np.ones(gt_img.shape[0]) * (num_domains - 1)
    #         lr_img = downgrade_in_seq(gt_img.clone(), target_domain, domains_interval)
    #
    #         with torch.no_grad():
    #             sr_img = gt_img.clone()
    #             previous_prediction = None
    #             inverse_momentum = 0
    #             for i in reversed(range(num_domains - 1)):
    #                 t = torch.ones(args.batch_size, device=device).long() * (i + 1)
    #                 if previous_prediction is None:
    #                     inverse_momentum_LR = 0
    #                 else:
    #                     inverse_momentum_LR = downgrade_in_seq(inverse_momentum.clone(), t, domains_interval)
    #
    #                 l, h = encoder(sr_img + inverse_momentum_LR, t)
    #                 result = decoder(l, t, h)
    #                 temp_result = result.clone()
    #                 if previous_prediction is None:
    #                     previous_prediction = temp_result.clone()
    #                 inverse_momentum += (previous_prediction - temp_result)
    #                 previous_prediction = temp_result.clone()
    #
    #                 sr_img = downgrade_in_seq(result, t - 1, domains_interval)
    #
    #         # vutils.save_image(torch.cat([
    #         #     lr_img.add(1).mul(0.5),
    #         #     result.add(1).mul(0.5),
    #         #     gt_img.add(1).mul(0.5)]),
    #         #     saved_image_folder + '/with_gt/%s.jpg' % imgname)
    #         vutils.save_image(result.add(1).mul(0.5),
    #                           saved_image_folder + '/result_only/%s.jpg' % imgname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DTLS')

    parser.add_argument('--path', type=str, default='world_real_lr_iii', #/hdda/Datasets/celeba/data1024x1024',
                        help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--output_path', type=str, default='DTLS_realLR_512_32_iv', help='Output path for the train results')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    # parser.add_argument('--name', type=str, default='32_1024_test_on_celeba_uneven', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=1, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--samples', type=int, default=4, help='number of samples to be SR')

    parser.add_argument('--ckpt', type=str, default="train_results/32_1024_retrain_universal_iic/models/500000.pth")
                                                    # "train_results/32_1024_retrain_universal_iic/models/500000.pth"
                                                    # "trained_weights/300000_32_1024_uneven_arti_fake.pth"
                                                    # "trained_weights/300000_universal.pth"
                                                    # "DTLS_super_resolution_32_1024_uneven_iii/models/200000.pth"
                                                    # trained_weights/DTLS_best_32_1024.pth

    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--input_folder', default=None, help='input folder for test')

    ### Args for DTLS ###
    # domains_interval =[512, 448, 384, 320, 256, 64, 16] #512
    # domains_interval =[256, 208, 160, 112, 64, 32, 16] # 256
    domains_interval = [1024, 896, 768, 512, 256, 128, 32]  # 1024
    #domains_interval = [1024, 768, 512, 256, 128, 64, 32]  # 1024

    args = parser.parse_args()

    inference(args, domains_interval)
