from __future__ import print_function

import argparse
import os
import platform
import time

import cv2
import h5py
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import DATASET_NAMES, TrainDataset, TestDataset, dataset_info
from losses import *
from model import DexiNed
from model_learnable_sigmoid import DexiNed_learnable_sigmoid

from utils import (save_image_batch_to_disk, visualize_result, visualize_result_ml_hypersim)

IS_LINUX = True if platform.system() == "Linux" else False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DexiNed trainer.')
    parser.add_argument('--choose_test_data', type=int, default=-1,
                        help='Already set the dataset for testing choice: 0 - 11')
    parser.add_argument('--choose_train_data', type=int, default=-1,
                        help='Already set the dataset for testing choice: 0 - 11')

    TEST_DATA = DATASET_NAMES[parser.parse_args().choose_test_data]  # max 11
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
    test_dir = test_inf['data_dir']
    is_testing = False  # current test _bdcnlossNew256-sd7-1.10.4p5

    # test related
    parser.add_argument('--is_testing', type=bool, default=is_testing, help='Script in testing mode.')
    parser.add_argument('--use_dataset', default=True, type=bool, help='test: dataset=True; single image=FALSE')
    parser.add_argument('--test_data', type=str, choices=DATASET_NAMES, default=TEST_DATA, help='Name of the dataset.')
    parser.add_argument('--test_list', type=str, default=test_inf['test_list'], help='Dataset sample indices list.')
    parser.add_argument('--test_img_height', type=int, default=test_inf['img_height'], help='Image height for testing.')
    parser.add_argument('--test_img_width', type=int, default=test_inf['img_width'], help='Image width for testing.')

    # Training settings
    TRAIN_DATA = DATASET_NAMES[parser.parse_args().choose_train_data]  # BIPED=0
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    train_dir = train_inf['data_dir']

    # training data
    parser.add_argument('--input_dir', type=str, default=train_dir,
                        help='the path to the directory with the input data.')
    parser.add_argument('--input_val_dir', type=str, default=test_dir,
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='the path to output the results.')
    parser.add_argument('--train_data', type=str, choices=DATASET_NAMES, default=TRAIN_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--train_list', type=str, default=train_inf['train_list'], help='Dataset sample indices list.')
    parser.add_argument('--img_width', type=int, default=train_inf['img_width'],
                        help='Image width for training.')  # BIPED 400 BSDS 352/320 MDBD 480
    parser.add_argument('--img_height', type=int, default=train_inf['img_height'],
                        help='Image height for training.')  # BIPED 480 BSDS 352/320
    parser.add_argument('--mean_pixel_values', default=[103.939, 116.779, 123.68, 137.86],
                        type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]

    # training detail
    parser.add_argument('--data_augmentation', type=bool, default=True,
                        help='(BOOL) whether or not to use data augmentation.')
    parser.add_argument('--double_img', type=bool, default=False,
                        help='True: use same 2 imgs changing channels')  # Just for test
    parser.add_argument('--channel_swap', default=[2, 1, 0], type=int)
    parser.add_argument('--crop_img', default=True, type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--resume', type=bool, default=True, help='use previous trained data')  # Just for test
    parser.add_argument('--checkpoint_data', type=str, default='14/14_model.pth',
                        help='Checkpoint path from which to restore model weights from.')
    parser.add_argument('--res_dir', type=str, default='result', help='Result directory')
    parser.add_argument('--log_interval_vis', type=int, default=50,
                        help='The number of batches to wait before printing test predictions.')
    parser.add_argument('--two_type', type=bool, default=False, help='Result directory')

    parser.add_argument('--epochs', type=int, default=34, metavar='N', help='Number of training epochs (default: 25).')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0., metavar='WD', help='weight decay (default: 1e-4) in F1=0')
    # parser.add_argument('--lr_stepsize', default=1e4, type=int, help='Learning rate step size.')
    parser.add_argument('--batch_size', type=int, default=8, metavar='B', help='the mini-batch size (default: 8)')
    parser.add_argument('--workers', default=1, type=int, help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard', type=bool, default=True, help='Use Tensorboard for logging.'),

    parser.add_argument('--use_learnable_sigmoid', type=bool, default=False, help='Use learnable sigmoid function.')
    args_out = parser.parse_args()
    return args_out


def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device,
                    log_interval_vis, tb_writer, args_in=None):
    imgs_res_folder = os.path.join(args_in.output_dir, 'current_res')
    os.makedirs(imgs_res_folder, exist_ok=True)

    # Put model in training mode
    model.train()
    # l_weight = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.1]  # for bdcn ori loss
    # before [0.6,0.6,1.1,1.1,0.4,0.4,1.3] [0.4,0.4,1.1,1.1,0.6,0.6,1.3],[0.4,0.4,1.1,1.1,0.8,0.8,1.3]
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]  # for bdcn loss theory 3 before the last 1.3 0.6-0..5
    # l_weight = [[0.05, 2.], [0.05, 2.], [0.05, 2.],
    #             [0.1, 1.], [0.1, 1.], [0.1, 1.],
    #             [0.01, 4.]]  # for cats loss
    loss_avg = []
    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW
        labels = sample_batched['labels'].to(device)  # BxCxHxW
        preds_list = model(images)
        # loss = sum([criterion(preds, labels, l_w, device) for preds, l_w in zip(preds_list, l_weight)])  # cats_loss
        loss = sum([criterion(preds, labels, l_w) / args_in.batch_size for preds, l_w in zip(preds_list, l_weight)])
        # bdcn_loss
        # loss = sum([criterion(preds, labels) for preds in preds_list])  #HED loss, rcf_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg.append(loss.item())
        if epoch == 0 and (batch_id == 100 and tb_writer is not None):
            tmp_loss = np.array(loss_avg).mean()
            tb_writer.add_scalar('loss', tmp_loss, epoch)

        if batch_id % 5 == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'
                  .format(epoch, batch_id, len(dataloader), loss.item()))
        if batch_id % log_interval_vis == 0:
            scene_name = None
            if args_in.train_data == 'ML-Hypersim':
                scene_name = sample_batched['scene_name'][2]

            res_data = []

            img = images.cpu().numpy()
            res_data.append(img[2])

            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt[2])

            # tmp_pred = tmp_preds[2,...]
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[2]
                # print(tmp.shape)
                tmp = torch.sigmoid(tmp)
                if args_in.train_data != 'ML-Hypersim':
                    tmp = tmp.unsqueeze(dim=0)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)

            if args_in.train_data == 'ML-Hypersim':
                vis_imgs = visualize_result_ml_hypersim(res_data, scene_name, arg=args_in)
            else:
                vis_imgs = visualize_result(res_data, arg=args_in)
            del tmp, res_data

            vis_imgs = cv2.resize(vis_imgs, (int(vis_imgs.shape[1] * 0.8), int(vis_imgs.shape[0] * 0.8)))
            img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3} File_name: {4}'.format(epoch, batch_id, len(dataloader),
                                                                                   loss.item(), scene_name)

            BLACK = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
            cv2.imwrite(os.path.join(imgs_res_folder, 'results.png'), vis_imgs)
    loss_avg = np.array(loss_avg).mean()
    return loss_avg


def validate_one_epoch(epoch, dataloader, model, device, output_dir, arg=None):
    # XXX This is not really validation, but testing
    print(epoch)
    # Put model in eval mode
    model.eval()

    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            # print(len(sample_batched['images']))
            images = sample_batched['images'].to(device)
            # labels = sample_batched['labels'].to(device)
            scene_name = sample_batched['scene_name']
            cam_name = sample_batched['cam_name']
            img_id = sample_batched['img_id']
            file_names = [f'{scene_name[i]}.{cam_name[i]}.{img_id[i]}.png' for i in range(len(scene_name))]
            # file_names = sample_batched['file_names']
            image_shape = [sample_batched['image_shape'][0].cpu().numpy()[0],
                           sample_batched['image_shape'][1].cpu().numpy()[0]]
            preds = model(images)
            # print(preds.shape)
            # exit()
            # print('pred shape', preds[0].shape)
            # print(preds[-1].shape)
            # print(output_dir)
            # print(file_names)
            # print(image_shape)
            save_image_batch_to_disk(preds[-1], output_dir, file_names, img_shape=image_shape, arg=arg)


def model_test(checkpoint_path, dataloader, model, device, output_dir, args_in):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not (args_in.test_data == "CLASSIC" or args_in.test_data == 'cabin'):
                _ = sample_batched['labels'].to(device)  # get the labels
            if args_in.test_data == 'ML-Hypersim':
                file_names = [sample_batched['scene_name'][0] + '_' + sample_batched['cam_name'][0] + '_' +
                              sample_batched['img_id'][0] + '.png']
                image_shape = sample_batched['image_shape']
            else:
                file_names = sample_batched['file_names']
                image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            # images = images[:, [2, 1, 0], :, :]
            start_time = time.time()
            preds = model(images)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args_in)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished in", args_in.test_data, "dataset. *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")


def verify_single(checkpoint_path, image_in, model, device, output_dir, args_in):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        images = image_in['data'].to(device)
        file_names = image_in['filename']
        image_shape = image_in['data'].shape
        print(f"input tensor shape: {images.shape}")
        # images = images[:, [2, 1, 0], :, :]
        start_time = time.time()
        preds = model(images)
        duration = time.time() - start_time
        save_image_batch_to_disk(preds,
                                 output_dir,
                                 file_names,
                                 image_shape,
                                 arg=args_in)
        torch.cuda.empty_cache()

    print("******** Testing finished in", args_in.test_data, "dataset. *****")
    print("Time spend: %f.4" % duration, "seconds")


def verifyPich(checkpoint_path, dataloader, model, device, output_dir, args_in):
    # a test model plus the interganged channels
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args_in.test_data == "CLASSIC":
                _ = sample_batched['labels'].to(device)  # get the labels
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            start_time = time.time()
            # images2 = images[:, [1, 0, 2], :, :]  #GBR
            images2 = images[:, [2, 1, 0], :, :]  # RGB
            preds = model(images)
            preds2 = model(images2)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk([preds, preds2],
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args_in, is_inchannel=True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished in", args_in.test_data, "dataset. *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")


def main(args_in):
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    # Tensorboard summary writer

    tb_writer = None
    training_dir = os.path.join(args_in.output_dir, args_in.train_data)
    os.makedirs(training_dir, exist_ok=True)
    checkpoint_path = os.path.join(args_in.output_dir, args_in.train_data, args_in.checkpoint_data)
    if args_in.tensorboard and not args_in.is_testing:
        # from tensorboardX import SummaryWriter  # previous torch version
        from torch.utils.tensorboard import SummaryWriter  # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir=training_dir)

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

    # Instantiate model and move it to the computing device
    if args_in.use_learnable_sigmoid:
        model = DexiNed_learnable_sigmoid(args_in).to(device)
    else:
        model = DexiNed(args_in).to(device)
    # model = nn.DataParallel(model)
    ini_epoch = 0
    dataloader_train = None
    dataloader_val = None
    if not args_in.is_testing:
        if args_in.resume:
            ini_epoch = 17
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        dataset_train = TrainDataset(args_in.input_dir,
                                     img_width=args_in.img_width,
                                     img_height=args_in.img_height,
                                     mean_bgr=args_in.mean_pixel_values[0:3] if len(
                                         args_in.mean_pixel_values) == 4 else args_in.mean_pixel_values,
                                     train_mode='train',
                                     arg=args_in
                                     )
        _ = dataset_train[0]  # get a data sample

        dataloader_train = DataLoader(dataset_train, batch_size=args_in.batch_size, shuffle=True,
                                      num_workers=args_in.workers)
    if args_in.use_dataset:
        dataset_val = TestDataset(args_in.input_val_dir,
                                  test_data=args_in.test_data,
                                  img_width=args_in.test_img_width,
                                  img_height=args_in.test_img_height,
                                  mean_bgr=args_in.mean_pixel_values[0:3] if len(
                                      args_in.mean_pixel_values) == 4 else args_in.mean_pixel_values,
                                  test_list=args_in.test_list,
                                  arg=args_in
                                  )
        _ = dataset_val[0]  # get a data sample
        dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args_in.workers)

    # Testing
    if args_in.is_testing:
        output_dir = os.path.join(args_in.res_dir, args_in.train_data + "2" + args_in.test_data)
        print(f"output_dir: {output_dir}")
        if args_in.use_dataset:
            if args_in.double_img:
                # predict twice an image changing channels, then mix those results
                verifyPich(checkpoint_path, dataloader_val, model, device, output_dir, args_in)
            else:
                model_test(checkpoint_path, dataloader_val, model, device, output_dir, args_in)
        else:
            img_data = h5py.File('/home/ubuntu/DexiNed/data/archive/Classic/frame.0000.depth_meters.hdf5', 'r')
            img_data = img_data['dataset'][:]
            img_test = {
                'data': torch.tensor(img_data),
                'filename': 'frame.0000.depth_meters'}
            verify_single(checkpoint_path, img_test, model, device, output_dir, args_in)
        return
    if args_in.two_type:
        criterion = multi_class
    else:
        criterion = bdcn_loss2
        # criterion = bdcn_loss_liu

        # criterion = mse_loss

    optimizer = optim.Adam(model.parameters(), lr=args_in.lr, weight_decay=args_in.wd)
    # lr_schd = lr_scheduler.StepLR(optimizer, step_size=args_in.lr_stepsize, gamma=args_in.lr_gamma)

    # Main training loop
    seed = 1021
    for epoch in range(ini_epoch, args_in.epochs):
        if epoch % 7 == 0:
            seed = seed + 1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        # Create output directories

        output_dir_epoch = os.path.join(args_in.output_dir, args_in.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args_in.test_data + '_res')
        os.makedirs(output_dir_epoch, exist_ok=True)
        os.makedirs(img_test_dir, exist_ok=True)

        avg_loss = train_one_epoch(epoch, dataloader_train, model, criterion, optimizer, device,
                                   args_in.log_interval_vis,
                                   tb_writer, args_in=args_in)
        validate_one_epoch(epoch, dataloader_val, model, device, img_test_dir, arg=args_in)

        # Save model after end of every epoch
        if args_in.use_learnable_sigmoid:
            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                       os.path.join(output_dir_epoch, 'learnable_sigmoid_{0}_model.pth'.format(epoch)))
        else:
            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                       os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        if tb_writer is not None:
            tb_writer.add_scalar('loss', avg_loss, epoch + 1)


if __name__ == '__main__':
    args = parse_args()
    main(args)
