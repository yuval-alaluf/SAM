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

	out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
	out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
	os.makedirs(out_path_results, exist_ok=True)
	os.makedirs(out_path_coupled, exist_ok=True)

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
							   opts=opts)
	dataloader = DataLoader(dataset,
							batch_size=opts.test_batch_size,
							shuffle=False,
							num_workers=int(opts.test_workers),
							drop_last=False)

	if opts.n_images is None:
		opts.n_images = len(dataset)

	global_time = []
	for age_transformer in age_transformers:
		print(f"Running on target age: {age_transformer.target_age}")
		global_i = 0
		for input_batch in tqdm(dataloader):
			if global_i >= opts.n_images:
				break
			with torch.no_grad():
				input_age_batch = [age_transformer(img.cpu()).to('cuda') for img in input_batch]
				input_age_batch = torch.stack(input_age_batch)
				input_cuda = input_age_batch.cuda().float()
				tic = time.time()
				result_batch = run_on_batch(input_cuda, net, opts)
				toc = time.time()
				global_time.append(toc - tic)

				for i in range(len(input_batch)):
					result = tensor2im(result_batch[i])
					im_path = dataset.paths[global_i]

					if opts.couple_outputs or global_i % 100 == 0:
						input_im = log_image(input_batch[i], opts)
						resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
						res = np.concatenate([np.array(input_im.resize(resize_amount)),
											  np.array(result.resize(resize_amount))], axis=1)
						age_out_path_coupled = os.path.join(out_path_coupled, age_transformer.target_age)
						os.makedirs(age_out_path_coupled, exist_ok=True)
						Image.fromarray(res).save(os.path.join(age_out_path_coupled, os.path.basename(im_path)))

					age_out_path_results = os.path.join(out_path_results, age_transformer.target_age)
					os.makedirs(age_out_path_results, exist_ok=True)
					image_name = os.path.basename(im_path)
					im_save_path = os.path.join(age_out_path_results, image_name)
					Image.fromarray(np.array(result.resize(resize_amount))).save(im_save_path)
					global_i += 1

	stats_path = os.path.join(opts.exp_dir, 'stats.txt')
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)

	with open(stats_path, 'w') as f:
		f.write(result_str)


def run_on_batch(inputs, net, opts):
	result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
	return result_batch


if __name__ == '__main__':
	run()
