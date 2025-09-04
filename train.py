import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import imgaug.augmenters as iaa
import imgaug as ia
import wandb

from diffaug import DiffAugment
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils import data
from models import weights_init, Encoder, Decoder, Discriminator
from operation import copy_G_params, load_params, get_dir, InfiniteSamplerWrapper

policy = "color,translation,cutout"

transform_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
trans = transforms.Compose(transform_list)

def downgrade_in_seq(image, target_size, domain_list):
    ori_size = image.shape[2]
    for j in range(image.shape[0]):
        image_a = image[j].clone().unsqueeze(0)
        for i in range(int(target_size[j])): # target_size = t
            image_a = F.interpolate(image_a, size=domain_list[i+1], mode='bicubic', antialias=True)
        image[j] = F.interpolate(image_a, size=ori_size, mode='bicubic', antialias=True).squeeze()
    return image

def downgrade_training(x, target_size, domain_list):
    org_size = domain_list[0]
    for j in range(x.shape[0]):
        scale_size = domain_list[target_size[j]]
        x[j] = random_augment(x[j], org_size, scale_size)
    return x

def random_augment(x, org_size, scale_size):
    """input single RGB PIL Image instance"""
    x = np.array(x.add(1).mul(255/2))
    x = np.uint8(x)
    # x = x[np.newaxis, :, :, :]
    aug_seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.OneOf([
            iaa.GaussianBlur((3, 15)),
            iaa.AverageBlur(k=(3, 15)),
            iaa.MedianBlur(k=(3, 15)),
            iaa.MotionBlur((5, 25))
        ])),
        iaa.Resize(scale_size, interpolation=ia.ALL),
        #iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5)),
        iaa.Sometimes(0.7, iaa.JpegCompression(compression=(10, 65))),
        iaa.Resize(org_size, interpolation=ia.ALL),
    ])
    aug_img = aug_seq(images=x)
    return torch.from_numpy(aug_img).mul(2/255).add(-1)


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.1), int(image_size*1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def train(args, domains_interval):
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    nlr = 0.00002
    use_cuda = True
    dataloader_workers = args.workers
    current_iteration = args.start_iter
    save_interval = args.save_interval
    saved_model_folder, saved_image_folder = get_dir(args)

    wandb.init(project="DTLS_1024", name=args.name)

    num_domains = len(domains_interval)
    
    if use_cuda:
        device = f"cuda:{args.cuda}"
    else:  device = torch.device("cpu")

    dataset = Dataset(data_root, im_size)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=False))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''

    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    encoder.apply(weights_init)
    decoder.apply(weights_init)
    discriminator.apply(weights_init)

    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)

    ema_encoder = copy_G_params(encoder)
    ema_decoder = copy_G_params(decoder)
    ema_discriminator = copy_G_params(discriminator)

    real_fixed_vector = next(dataloader).to(device)
    target_domain = np.ones(real_fixed_vector.shape[0]) * (num_domains-1)

    fixed_vector = downgrade_in_seq(real_fixed_vector.clone(), target_domain, domains_interval)

    BCE_loss = nn.BCEWithLogitsLoss()

    optimizer_enc = optim.AdamW(encoder.parameters(), lr=nlr, betas=(0.9, 0.999))
    optimizer_dec = optim.AdamW(decoder.parameters(), lr=nlr, betas=(0.9, 0.999))
    optimizer_dis = optim.AdamW(discriminator.parameters(), lr=nlr, betas=(0.9,0.999))

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        encoder.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['enc'].items()})
        decoder.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['dec'].items()})
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt

    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)
        current_batch_size = real_image.size(0)

        target_domain = np.random.choice(num_domains-1, current_batch_size, p=[0.025, 0.025, 0.05, 0.1, 0.2, 0.6]) + 1   # uneven

        t = torch.ones(current_batch_size).long() * torch.tensor(target_domain)
        t = t.to(device)

        downgraded_image = downgrade_training(real_image.clone(), target_domain, domains_interval).to(device)
        real_image = real_image.to(device)
        latent_img, h = encoder(downgraded_image, t)
        recon_img = decoder(latent_img, t, h)


        discriminator.zero_grad()
        target_score = discriminator(real_image)
        dis_err_real = BCE_loss(target_score, torch.ones_like(target_score))
        dis_err_real.backward()

        chance = np.random.rand(1)
        
        ## Artificial fake image
        if chance < 0.5:
            current_score = discriminator(recon_img.detach())
        else:
            fake_img = DiffAugment(real_image.clone(),policy)
            current_score = discriminator(fake_img)
            
        dis_err_fake = BCE_loss(current_score, torch.zeros_like(current_score))
        dis_err_fake.backward()
        optimizer_dis.step()

        encoder.zero_grad()
        decoder.zero_grad()

        err_mse = ((recon_img - real_image)**2).mean()
        updated_score = discriminator(recon_img)
        err_domain = BCE_loss(updated_score, torch.ones_like(updated_score)) * 1e-3

        (err_mse + err_domain).backward()
        optimizer_enc.step()
        optimizer_dec.step()

        wandb.log({"MSE loss": err_mse.item(), "latent realness loss": err_domain.item(),
                   "Discriminator fake loss": dis_err_fake.item(), "Discriminator real loss": dis_err_real.item()})

        ### Averaging parameters
        for p, avg_p in zip(encoder.parameters(), ema_encoder):
            avg_p.mul_(0.999).add_(0.001 * p.data)
        for p, avg_p in zip(decoder.parameters(), ema_decoder):
            avg_p.mul_(0.999).add_(0.001 * p.data)
        for p, avg_p in zip(discriminator.parameters(), ema_discriminator):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % (save_interval*10) == 0:
            load_params(encoder, ema_encoder)
            load_params(decoder, ema_decoder)
            load_params(discriminator, ema_discriminator)

            with torch.no_grad():
                result = fixed_vector
                result_list = result

                previous_prediction = None
                inverse_momentum = 0
                for i in reversed(range(1, num_domains)):
                    t = torch.ones(current_batch_size, device=device).long() * i
                    print(t)

                    if previous_prediction is None:
                        inverse_momentum_LR = 0
                    else:
                        inverse_momentum_LR = downgrade_in_seq(inverse_momentum.clone(), t, domains_interval)

                    l, h = encoder(result + inverse_momentum_LR, t)
                    result = decoder(l, t, h)
                    temp_result = result.clone()
                    if previous_prediction is None:
                        previous_prediction = temp_result.clone()
                    inverse_momentum += (previous_prediction - temp_result)
                    previous_prediction = temp_result.clone()
                    if i > 1:
                        result = downgrade_in_seq(result, t - 1, domains_interval)
                        result_list = torch.cat((result_list, result))
                    else:
                        break

            vutils.save_image(torch.cat([
                    fixed_vector.add(1).mul(0.5),
                    result.add(1).mul(0.5),
                    real_fixed_vector.add(1).mul(0.5)]),
                saved_image_folder+'/%d.jpg'%iteration, nrow=real_image.shape[0])

            vutils.save_image(result_list.add(1).mul(0.5),
                saved_image_folder+'/domains_%d.jpg'%iteration, nrow=real_image.shape[0])

            vutils.save_image(torch.cat([
                downgraded_image.add(1).mul(0.5),
                recon_img.add(1).mul(0.5),
                real_image.add(1).mul(0.5)]),
                saved_image_folder + '/training_%d.jpg' % iteration, nrow=real_image.shape[0])

            wandb.log({"Checkpoint result": wandb.Image(saved_image_folder+'/%d.jpg'%iteration)})
            wandb.log({"Result in domains": wandb.Image(saved_image_folder+'/domains_%d.jpg'%iteration)})

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            torch.save({'enc':encoder.state_dict(), 'dec':decoder.state_dict()}, saved_model_folder+'/%d.pth'%iteration)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='images1024x1024', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--output_path', type=str, default='./', help='Output path for the train results')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='DTLS_1024_FFHQ', help='experiment name')
    parser.add_argument('--iter', type=int, default=300001, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=2, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--save_interval', type=int, default=10000, help='number of iterations to save model')
    
    domains_interval =[1024, 896, 768, 512, 256, 128, 32] # 1024

    args = parser.parse_args()
    print(args)

    train(args, domains_interval)
