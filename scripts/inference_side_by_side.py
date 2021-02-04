from argparse import Namespace
import os
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im, log_image
from options.test_options import TestOptions
from models.psp import pSp


def run():
	test_opts = TestOptions().parse()

	out_path_results = os.path.join(test_opts.exp_dir, 'inference_side_by_side')
	os.makedirs(out_path_results, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()

	age_transformers = [AgeTransformer(target_age=age) for age in opts.target_age.split(',')]

	print(f'Loading dataset for {opts.dataset_type}')
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	dataset = InferenceDataset(root=opts.data_path,
							   transform=transforms_dict['transform_inference'],
							   opts=opts,
							   return_path=True)
	dataloader = DataLoader(dataset,
							batch_size=opts.test_batch_size,
							shuffle=False,
							num_workers=int(opts.test_workers),
							drop_last=False)

	if opts.n_images is None:
		opts.n_images = len(dataset)

	global_time = []
	global_i = 0
	for input_batch, image_paths in tqdm(dataloader):
		if global_i >= opts.n_images:
			break
		batch_results = {}
		for idx, age_transformer in enumerate(age_transformers):
			with torch.no_grad():
				input_age_batch = [age_transformer(img.cpu()).to('cuda') for img in input_batch]
				input_age_batch = torch.stack(input_age_batch)
				input_cuda = input_age_batch.cuda().float()
				tic = time.time()
				result_batch = run_on_batch(input_cuda, net, opts)
				toc = time.time()
				global_time.append(toc - tic)

				resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
				for i in range(len(input_batch)):
					result = tensor2im(result_batch[i])
					im_path = image_paths[i]
					input_im = log_image(input_batch[i], opts)
					if im_path not in batch_results.keys():
						batch_results[im_path] = np.array(input_im.resize(resize_amount))
					batch_results[im_path] = np.concatenate([batch_results[im_path],
															 np.array(result.resize(resize_amount))],
															axis=1)

		for im_path, res in batch_results.items():
			image_name = os.path.basename(im_path)
			im_save_path = os.path.join(out_path_results, image_name)
			Image.fromarray(np.array(res)).save(im_save_path)
			global_i += 1


def run_on_batch(inputs, net, opts):
	result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
	return result_batch


if __name__ == '__main__':
	run()
