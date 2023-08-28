import os
import sys
from collections import defaultdict
from statistics import mean

import cv2
# library for fine-tuning
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import draw
from torch.nn.functional import threshold, normalize

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from tqdm import tqdm


class Train:
    def __init__(self, train_dir='./dataset/medical_sinusitis/polygon/train', val_dir='./', test_dir='./',
                 dataset_type='polygon', device='0'):
        self.losses = []
        self.train_dir = train_dir
        self.image_list = {}
        self.dataset_type = dataset_type
        self.device = device

    def check_device(self):
        device = self.device
        if torch.cuda.is_available():
            cuda_number = int(torch.cuda.device_count())
            if int(device) < cuda_number:
                print(torch.cuda.get_device_name(device=int(device)))
        else:
            print('Warning! No devices detected. check your device or cuda toolkit!')
            sys.exit()

    def dataset(self):
        file_ex = '.txt'
        label_list = [file.split('.txt')[0] for file in os.listdir(self.train_dir + 'labels/') if
                      file.endswith(file_ex)]
        coords_list = {}
        clss = {}

        for file in label_list:
            with open(self.train_dir + '/labels/' + file + '.txt', 'r') as txtFile:
                coords_list[file] = list(map(str, (txtFile.read().split('\n'))))

        for file_path in coords_list:
            coord_list = list()
            for n, j in enumerate(coords_list[file_path]):
                coord_list.append(list(map(float, j.split())))
                clss[file_path] = coord_list[n].pop(0)
            self.image_list[file_path] = coord_list

    def encoder(self, model_type='vit_h', checkpoint='sam_vit_h_4b8939.pth', lr=1e-4, wd=0.1, epochs=5):
        def polygon2bbox(input_polygons):
            output_polygons = input_polygons.copy()
            for n, polygon in enumerate(input_polygons):
                x1 = polygon[0]
                y1 = polygon[1]
                x2 = polygon[0]
                y2 = polygon[1]
                for xy, instance in enumerate(polygon):
                    if xy % 2 == 0:
                        if x1 > instance:
                            x1 = instance
                        if x2 < instance:
                            x2 = instance
                    else:
                        if y1 > instance:
                            y1 = instance
                        if y2 < instance:
                            y2 = instance
                output_polygons[n] = [x1, y1, x2, y2]
            return output_polygons

        import time

        device = self.device
        image_list = self.image_list
        train_dir = self.train_dir
        if self.dataset_type == 'polygon':
            bbox_coords = {}
            for path in self.image_list:
                bbox_coords[path] = polygon2bbox(self.image_list[path])

        ground_truth_masks = {}
        for k in image_list.keys():
            gt_size = cv2.imread(os.path.join(train_dir, 'images', k + '.jpg'), cv2.IMREAD_GRAYSCALE).shape[0:2]
            mask_list = list()
            for num, coords in enumerate(image_list[k]):
                temp_list = list()
                for i, coord in enumerate(range(0, len(coords), 2)):
                    temp_list.append((int(coords[i * 2 + 1] * gt_size[1]), int(coords[i * 2] * gt_size[0])))
                mask_list.append(draw.polygon2mask(gt_size, temp_list))
            ground_truth_masks[k] = mask_list

        # code for fine-tuning
        # checkpoint = 'sam_vit_b_01ec64.pth'
        # checkpoint = 'sam_vit_h_4b8939.pth'
        # checkpoint = 'sam_vit_h_4b8939_fine_medical_5_0.0001all.pth'
        self.check_device()
        device = 'cuda:' + device

        sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        sam_model.to(device)
        sam_model.train()

        # Preprocess the images
        transformed_data = defaultdict(dict)
        for k in bbox_coords.keys():
            image = cv2.imread(self.train_dir + 'images/' + k + '.jpg')
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            input_image = transform.apply_image(image)
            input_image_torch = torch.as_tensor(input_image, device=device)
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            input_image = sam_model.preprocess(transformed_image)
            original_image_size = image.shape[:2]
            input_size = tuple(transformed_image.shape[-2:])

            transformed_data[k]['image'] = input_image
            transformed_data[k]['input_size'] = input_size
            transformed_data[k]['original_image_size'] = original_image_size

        print('Learning rate:', lr)
        print('wd:', wd)
        optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

        loss_fn = torch.nn.MSELoss()

        # Run Fine-tuning
        print('Training epochs:', epochs)
        pbar = tqdm(bbox_coords.keys())
        for epoch in range(epochs):
            epoch_losses = []
            print(('%11s' * 2 + '%11s' * 1) % ('epoch', 'gpu_memory', 'loss'))
            for k in pbar:
                input_image = transformed_data[k]['image'].to(device)
                input_size = transformed_data[k]['input_size']
                original_image_size = transformed_data[k]['original_image_size']
                for i in range(len(bbox_coords[k])):
                    # for i in range(1):
                    # No grad here as we don't want to optimise the encoders
                    with torch.no_grad():
                        image_embedding = sam_model.image_encoder(input_image)

                        prompt_box = np.array(bbox_coords[k][i])
                        box = transform.apply_boxes(prompt_box, original_image_size)
                        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                        box_torch = box_torch[None, :]

                        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )
                    low_res_masks, iou_predictions = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(
                        device)
                    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

                    gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k][i], (
                        1, 1, ground_truth_masks[k][i].shape[0], ground_truth_masks[k][i].shape[1]))).to(
                        device)
                    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32).to(device)

                    loss = loss_fn(binary_mask, gt_binary_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%11s' * 2 + '%11.4g' * 1) %
                                         (f'{epoch+1}/{epochs}', mem, loss.item()))
            self.losses.append(epoch_losses)
            print(f'Mean loss: {mean(epoch_losses)}')

        now = time.localtime()
        now = ("%04d%02d%02d_%02d%02d%02d" %
               (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

        path = './'
        file_list = os.listdir(path)
        if 'run' not in file_list:
            os.mkdir('run')
            print(f'There\'s no run directory. A run directory was made')
        file_list = os.listdir(path + 'run/')
        if now not in file_list:
            os.mkdir('./run/' + now)
            print(f'There\'s no {now} directory. An {now} directory was made')
        file_list = os.listdir(path + 'run/' + now)
        if 'checkpoint' not in file_list:
            os.mkdir('./run/' + now + '/checkpoint')
            print(f'There\'s no checkpoint directory. A checkpoint directory was made')

        torch.save(sam_model.state_dict(),
                   './run/' + now + '/checkpoint/sam_vit_h_4b8939_fine_medical_' + str(epochs) + '_' + str(lr) + '.pth')
        mean_losses = [mean(x) for x in self.losses]

        plt.plot(list(range(len(mean_losses))), mean_losses)
        plt.title('Mean epoch loss')
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')

        file_list = os.listdir(path + 'run/' + now)
        if 'train_result' not in file_list:
            os.mkdir('./run/' + now + '/train_result')
            print(f'There\'s no train_result directory. A train_result directory was made')

        plt.savefig('./run/' + now + '/train_result/sam_vit_h_4b8939_fine_medical_' + str(epochs) + '_' + str(lr) +
                    '.png')
        plt.show()
