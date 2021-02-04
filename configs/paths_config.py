dataset_paths = {
    'celeba_test': '/nfs/datasets/segmentation/celebAMask/CelebAMask-HQ/test_img',
    'ffhq': '/nfs/datasets/segmentation/ffhq/images256x256',
}

model_paths = {
    'pretrained_psp': '/nfs/weights/pix2profile/ablations/ffhq_gradual_archi_292500.pt',
    'ir_se50': '/nfs/weights/pix2profile/model_ir_se50.pth',
    'stylegan_ffhq': '/nfs/weights/pix2profile/stylegan2-ffhq-config-f.pt',
    'shape_predictor': '/nfs/weights/pix2profile/shape_predictor_68_face_landmarks.dat',
    'age_predictor': '/nfs/outputs/alae_segmentation/train_age_estimation/no_resized_crop/checkpoints/epoch033_0.47652_4.6758.pth'
}

# dataset_paths = {
#     'celeba_test': '',
#     'ffhq': '',
# }
#
# model_paths = {
#     'pretrained_psp_encoder': 'pretrained_models/psp_ffhq_encode.pt',
#     'ir_se50': 'pretrained_models/model_ir_se50.pth',
#     'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
#     'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
#     'age_predictor': 'pretrained_models/dex_age_classifier.pth'
# }
