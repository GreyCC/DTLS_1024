# Domain Transfer in Latent Space (DTLS) Wins on Image Super-Resolution - a Non-Denoising Model

This is the program page of our latest DTLS approach for image super-resolution.

Chun-Chuen Hui, Wan-Chi Siu, and Ngai-Fong Law, "Domain Transfer in Latent Space (DTLS) Wins on Image Super-Resolution - A Non-Denoising Model", 2025. 

## System requirments


## Dataset
First, you need to download the dataset for training, in our experimental session, we utilize FFHQ dataset, you can download the dataset directly from this link:
https://drive.google.com/file/d/1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv/view?usp=drive_link[https://drive.google.com/file/d/1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv/view?usp=drive_link]

After finish the download of the dataset, you can move the dataset folder inside the folder of DTLS_1024 (This program code).

## Training
Train the model by following the command lines below
```
python train.py --path FFHQ1024 --cuda 0 --name DTLS_1024_FFHQ
```
Details:

```
--path [should be filled in the directory to the FFHQ dataset]

--cuda [if your company equipped with 2 or more GPU, you can choose other than zero e.g. 1 if 2 gpu equipped]

--name [folder name to store the result of this training]
```

You should find the training result within the folder: "train_results/[--name]"

Depending on your computing power, training time may vary from 1 to 3 days

## Recall
After the training you can run the following command to super-resolute face images for evaluation.

We have provide samples of natural low-resolution face images inside folder 'NLQ_Faces' for evaluation.

```
python eval.py --path 'NLQ_Faces' --cuda 0 --output_path 'DTLS_1024_NLR_test' --ckpt 'train_results/DTLS_1024_FFHQ/models/300000.pth'
```

Details:
```
--path [path to our neatureal low-resolution face images; or path to other input folder included LR faces]

--cuda [if your company equipped with 2 or more GPU, you can choose other than zero e.g. 1 if 2 gpu equipped]

--output_path [folder name to store the result of this evaluation]

--ckpt [path to the saved weights of the U-Net you just trained in previous section]
```
