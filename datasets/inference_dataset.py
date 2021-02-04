from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class InferenceDataset(Dataset):

	def __init__(self, root=None, paths_list=None, opts=None, transform=None, return_path=False):
		if paths_list is None:
			self.paths = sorted(data_utils.make_dataset(root))
		else:
			self.paths = data_utils.make_dataset_from_paths_list(paths_list)
		self.transform = transform
		self.opts = opts
		self.return_path = return_path

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
		if self.transform:
			from_im = self.transform(from_im)
		if self.return_path:
			return from_im, from_path
		else:
			return from_im
