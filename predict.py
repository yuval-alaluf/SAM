import tempfile
from argparse import Namespace
import dlib
import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from cog import BasePredictor, Path, Input

from datasets.augmentations import AgeTransformer
from models.psp import pSp
from scripts.align_all_parallel import align_face
from utils.common import tensor2im


class Predictor(BasePredictor):
    def setup(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        model_path = "pretrained_models/sam_ffhq_aging.pt"
        ckpt = torch.load(model_path, map_location="cpu")

        opts = ckpt["opts"]
        opts["checkpoint_path"] = model_path
        opts["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.opts = Namespace(**opts)

    def predict(
            self,
            image: Path = Input(
                description="facial image",
            ),
            target_age: str = Input(
                description="age of the output image, when choose 'default' "
                            "a gif for age from 0, 10, 20,...,to 100 will be displayed",
            ),
    ) -> Path:
        net = pSp(self.opts)
        net.eval()
        if torch.cuda.is_available():
            net.cuda()

        # align image
        aligned_image = run_alignment(str(image))
        aligned_image.resize((256, 256))

        input_image = self.transform(aligned_image)

        if target_age == "default":
            target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        else:
            target_ages = target_age.split(',')
            target_ages = [int(age) for age in target_ages]
        
        age_transformers = [AgeTransformer(target_age=age) for age in target_ages]

        results = np.array(aligned_image.resize((1024, 1024)))
        all_imgs = []
        for age_transformer in age_transformers:
            print(f"Running on target age: {age_transformer.target_age}")
            with torch.no_grad():
                input_image_age = [age_transformer(input_image.cpu()).to("cuda")]
                input_image_age = torch.stack(input_image_age)
                result_tensor = run_on_batch(input_image_age, net)[0]
                result_image = tensor2im(result_tensor)
                all_imgs.append(result_image)
                results = np.concatenate([results, result_image], axis=1)

        if target_age == "default" or len(age_transformers) > 1:
            out_path = Path(tempfile.mkdtemp()) / "output.gif"
            imageio.mimwrite(str(out_path), all_imgs, duration=0.3)
        else:
            out_path = Path(tempfile.mkdtemp()) / "output.png"
            imageio.imwrite(str(out_path), all_imgs[0])
        return out_path


def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch
