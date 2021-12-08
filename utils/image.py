import os

import cv2
import h5py
import numpy as np
import torch
import kornia as kn
import pandas as pd


def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """
    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def save_image_batch_to_disk(tensor, output_dir, file_names, img_shape=None, arg=None, is_inchannel=False):
    os.makedirs(output_dir, exist_ok=True)
    if not arg.is_testing:
        assert len(tensor.shape) == 4, tensor.shape
        img_shape = np.array(img_shape)
        for tensor_image, file_name in zip(tensor, file_names):
            image_vis = kn.utils.tensor_to_image(
                torch.sigmoid(tensor_image))  # [..., 0]
            image_vis = (255.0 * (1.0 - image_vis)).astype(np.uint8)
            output_file_name = os.path.join(output_dir, file_name)
            image_vis = cv2.resize(image_vis, dsize=(img_shape[1], img_shape[0]))
            print(output_file_name)
            assert cv2.imwrite(output_file_name, image_vis)
    else:
        if is_inchannel:

            tensor, tensor2 = tensor
            fuse_name = 'fusedCH'
            av_name = 'avgCH'
            is_2tensors = True
            edge_maps2 = []
            for i in tensor2:
                tmp = torch.sigmoid(i).cpu().detach().numpy()
                edge_maps2.append(tmp)
            tensor2 = np.array(edge_maps2)
        else:
            fuse_name = 'fused'
            av_name = 'avg'
            tensor2 = None
            tmp_img2 = None

        output_dir_f = os.path.join(output_dir, fuse_name)
        output_dir_a = os.path.join(output_dir, av_name)
        os.makedirs(output_dir_f, exist_ok=True)
        os.makedirs(output_dir_a, exist_ok=True)

        # 255.0 * (1.0 - em_a)
        edge_maps = []
        for i in tensor:
            tmp = torch.sigmoid(i).cpu().detach().numpy()
            # tmp = i.cpu().detach().numpy()
            edge_maps.append(tmp)
        tensor = np.array(edge_maps)
        # print(f"tensor shape: {tensor.shape}")

        image_shape = [x.cpu().detach().numpy() for x in img_shape]
        # (H, W) -> (W, H)
        image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

        assert len(image_shape) == len(file_names)

        idx = 0
        for i_shape, file_name in zip(image_shape, file_names):
            tmp = tensor[:, idx, ...]
            tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
            # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
            tmp = np.squeeze(tmp)
            tmp2 = np.squeeze(tmp2) if tensor2 is not None else None

            # Iterate our all 7 NN outputs for a particular image
            preds = []
            for i in range(tmp.shape[0]):
                tmp_img = tmp[i]
                tmp_img = np.uint8(image_normalization(tmp_img))
                tmp_img = cv2.bitwise_not(tmp_img)
                # tmp_img[tmp_img < 0.0] = 0.0
                # tmp_img = 255.0 * (1.0 - tmp_img)
                if tmp2 is not None:
                    tmp_img2 = tmp2[i]
                    tmp_img2 = np.uint8(image_normalization(tmp_img2))
                    tmp_img2 = cv2.bitwise_not(tmp_img2)

                # Resize prediction to match input image size
                if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                    tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
                    tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None

                if tmp2 is not None:
                    tmp_mask = np.logical_and(tmp_img > 128, tmp_img2 < 128)
                    tmp_img = np.where(tmp_mask, tmp_img2, tmp_img)
                    preds.append(tmp_img)

                else:
                    preds.append(tmp_img)

                if i == 6:
                    fuse = tmp_img
                    fuse = fuse.astype(np.uint8)
                    if tmp_img2 is not None:
                        fuse2 = tmp_img2
                        fuse2 = fuse2.astype(np.uint8)
                        # fuse = fuse-fuse2
                        fuse_mask = np.logical_and(fuse > 128, fuse2 < 128)
                        fuse = np.where(fuse_mask, fuse2, fuse)

                        # print(fuse.shape, fuse_mask.shape)

            # Get the mean prediction of all the 7 outputs
            average = np.array(preds, dtype=np.float32)
            average = np.uint8(np.mean(average, axis=0))
            output_file_name_f = os.path.join(output_dir_f, file_name)
            output_file_name_a = os.path.join(output_dir_a, file_name)
            cv2.imwrite(output_file_name_f, fuse)
            cv2.imwrite(output_file_name_a, average)

            idx += 1


def restore_rgb(config, I, restore_rgb=False):
    """
    :param config: [args.channel_swap, args.mean_pixel_value]
    :param I: an image or a set of images
    :return: an image or a set of images restored
    """

    if len(I) > 3 and not type(I) == np.ndarray:
        I = np.array(I)
        I = I[:, :, :, 0:3]
        n = I.shape[0]
        for i in range(n):
            x = I[i, ...]
            x = np.array(x, dtype=np.float32)
            x += config[1]
            if restore_rgb:
                x = x[:, :, config[0]]
            x = image_normalization(x)
            I[i, :, :, :] = x
    elif len(I.shape) == 3 and I.shape[-1] == 3:
        I = np.array(I, dtype=np.float32)
        I += config[1]
        if restore_rgb:
            I = I[:, :, config[0]]
        I = image_normalization(I)
    else:
        print("Sorry the input data size is out of our configuration")
    return I


def visualize_result(imgs_list, arg):
    """
    data 2 image in one matrix
    :param imgs_list: a list of prediction, gt and input data
    :param arg:
    :return: one image with the whole of imgs_list data
    """
    n_imgs = len(imgs_list)
    data_list = []
    for i in range(n_imgs):
        tmp = imgs_list[i]
        # print(tmp.shape)
        if tmp.shape[0] == 3:
            tmp = np.transpose(tmp, [1, 2, 0])
            tmp = restore_rgb([arg.channel_swap, arg.mean_pixel_values[:3]], tmp)
            tmp = np.uint8(image_normalization(tmp))
        else:
            tmp = np.squeeze(tmp)
            if len(tmp.shape) == 2:
                tmp = np.uint8(image_normalization(tmp))
                tmp = cv2.bitwise_not(tmp)
                tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
            else:
                tmp = np.uint8(image_normalization(tmp))
        data_list.append(tmp)
        # print(i,tmp.shape)
    img = data_list[0]
    if n_imgs % 2 == 0:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * (n_imgs // 2) + ((n_imgs // 2 - 1) * 5), 3))
    else:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * ((1 + n_imgs) // 2) + ((n_imgs // 2) * 5), 3))
        n_imgs += 1

    k = 0
    imgs = np.uint8(imgs)
    i_step = img.shape[0] + 10
    j_step = img.shape[1] + 5
    for i in range(2):
        for j in range(n_imgs // 2):
            if k < len(data_list):
                imgs[i * i_step:i * i_step + img.shape[0], j * j_step:j * j_step + img.shape[1], :] = data_list[k]
                k += 1
            else:
                pass
    return imgs


def visualize_result_ml_hypersim(imgs_list, scene_name, arg):
    """
    data 2 image in one matrix
    :param imgs_list: a list of prediction, gt and input data
    :param arg:
    :return: one image with the whole of imgs_list data
    """
    n_imgs = len(imgs_list)
    data_list = []
    for i in range(n_imgs):
        tmp = imgs_list[i]
        # print(tmp.shape)
        if tmp.shape[0] == 3:
            if i == 0:  # rgb
                tmp = np.transpose(tmp, [1, 2, 0])
                tmp = restore_rgb([arg.channel_swap, arg.mean_pixel_values[:3]], tmp)
                tmp = np.uint8(image_normalization(tmp))
                data_list.append(tmp)
            else:
                tmp = np.transpose(tmp, [1, 2, 0])
                tmp0 = np.zeros_like(tmp[:, :, 1])
                tmp1 = np.uint8(image_normalization(tmp[:, :, 1]))
                tmp2 = np.uint8(image_normalization(tmp[:, :, 2]))
                tmp1[np.where(tmp1 < 150)] = 0
                tmp2[np.where(tmp2 < 150)] = 0
                tmp = np.stack([tmp2, tmp0, tmp1], axis=2)
                data_list.append(tmp)
        elif tmp.shape[0] == 4:
            tmp = np.around(tmp).astype(int)
            tmp = np.transpose(tmp, [1, 2, 0])
            geometry_edge_img = tmp[:, :, 0:3]
            geometry_edge_img = np.stack(
                [geometry_edge_img[:, :, 2], geometry_edge_img[:, :, 1], geometry_edge_img[:, :, 0]], axis=2).astype(
                np.uint8)
            render_entity_id_img = tmp[:, :, 3]

            # images_dir = os.path.join(arg.input_dir, scene_name[i-1], "images")
            detail_dir = os.path.join(arg.input_dir, scene_name, '_detail')
            in_mesh_objects_segmentation_file = os.path.join(detail_dir, "mesh", "mesh_objects_sii.hdf5")
            in_metadata_segmentation_colors_file = os.path.join(detail_dir, "mesh",
                                                                "metadata_semantic_instance_colors.hdf5")
            # in_render_entity_id_hdf5_dir = os.path.join(images_dir, 'scene_' + cam_name[i-1] + "_geometry_hdf5")
            in_metadata_nodes_file = os.path.join(detail_dir, "metadata_nodes.csv")

            df_nodes = pd.read_csv(in_metadata_nodes_file)
            node_ids_ = df_nodes["node_id"].to_numpy()
            # node_ids = np.r_[-1, node_ids_]  # numpy magic function, used to concatenate new elements or new arrays
            # node_id_max = node_ids_.shape[0]

            node_ids_to_mesh_object_ids_ = df_nodes["object_id"].to_numpy()
            node_ids_to_mesh_object_ids = np.r_[-1, node_ids_to_mesh_object_ids_]

            with h5py.File(in_mesh_objects_segmentation_file, "r") as f:
                mesh_object_ids_to_segmentation_indices = np.matrix(f["dataset"][:, 0]).A1.astype(np.int32)
                # mesh_object_id_max = mesh_object_ids_to_segmentation_indices.shape[0] - 1  # 说明id从0开始

            with h5py.File(in_metadata_segmentation_colors_file, "r") as f:
                segmentation_ids_to_segmentation_colors = f["dataset"][:]
                # segmentation_id_max = segmentation_ids_to_segmentation_colors.shape[0] - 1

            # in_filename = f"frame.{img_id}.render_entity_id.hdf5"
            # in_render_entity_id_hdf5_file = os.path.join(in_render_entity_id_hdf5_dir, in_filename)

            # with h5py.File(in_render_entity_id_hdf5_file, "r") as f:
            #     render_entity_id_img = f["dataset"][:].astype(np.int32)

            mesh_object_id_img = np.ones_like(render_entity_id_img) * -1
            segmentation_id_img = np.ones_like(render_entity_id_img) * -1
            segmentation_color_img = np.zeros((render_entity_id_img.shape[0], render_entity_id_img.shape[1], 3),
                                              dtype=np.uint8)

            mesh_object_id_img[render_entity_id_img != -1] = node_ids_to_mesh_object_ids[
                render_entity_id_img[render_entity_id_img != -1]]
            segmentation_id_img[mesh_object_id_img != -1] = mesh_object_ids_to_segmentation_indices[
                mesh_object_id_img[mesh_object_id_img != -1]]
            segmentation_color_img[segmentation_id_img != -1] = segmentation_ids_to_segmentation_colors[
                segmentation_id_img[segmentation_id_img != -1]]
            # from PIL import Image
            # seg_show = Image.fromarray(segmentation_color_img)
            # seg_show.show()
            #
            # # geometry_edge_img = np.zeros((geometry_edge.shape[0], geometry_edge.shape[1], 3), dtype=np.uint8)
            # # geometry_edge_img[np.where(geometry_edge == 2)] = np.array([255, 0, 255])
            # # geometry_edge_img[np.where(geometry_edge == 3)] = np.array([255, 255, 0])
            #
            # geo_show = Image.fromarray(geometry_edge_img)
            # geo_show.show()
            data_list.append(geometry_edge_img)
            data_list.append(segmentation_color_img)
        else:  #
            tmp = np.squeeze(tmp)
            if arg.two_type:
                tmp_new = np.zeros([tmp.shape[0], tmp.shape[1], 3]).astype(np.uint8)
                tmp_new[np.where(tmp == 1)] = np.array([0, 0, 255])
                tmp_new[np.where(tmp == 2)] = np.array([255, 0, 0])
                tmp = tmp_new
            else:
                if len(tmp.shape) == 2:
                    tmp = np.uint8(image_normalization(tmp))
                    tmp = cv2.bitwise_not(tmp)
                    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
                else:
                    tmp = np.uint8(image_normalization(tmp))
            data_list.append(tmp)

        # print(i,tmp.shape)

    # 多尺度融合和，将原始图片中多张图片进行融合. 错误了，就是单纯的图片拼接而已
    img = data_list[0]
    n_imgs = len(data_list)  # take the color image
    if n_imgs % 2 == 0:  # 如果图片总数是2的倍数
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * (n_imgs // 2) + ((n_imgs // 2 - 1) * 5), 3))
    else:
        # 创建一个图片矩阵，宽度是原始图片宽度乘2加10，宽度是原始图片个数加一除以2的整数部分，（也就是乘5）加上图片数量处以2（4）
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * ((1 + n_imgs) // 2) + ((n_imgs // 2) * 5), 3))
        n_imgs += 1

    k = 0
    imgs = np.uint8(imgs)
    i_step = img.shape[0] + 10
    j_step = img.shape[1] + 5
    for i in range(2):
        for j in range(n_imgs // 2):
            if k < len(data_list):
                # print(k)      # debug
                imgs[i * i_step:i * i_step + img.shape[0], j * j_step:j * j_step + img.shape[1], :] = data_list[k]
                k += 1
            else:
                pass

    return imgs
