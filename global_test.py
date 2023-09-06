import os
import cv2
import time
import torch
import shutil
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from Reconstruction_3D import models
from Reconstruction_3D import datasets
from Reconstruction_3D.loggers import loggers
import Reconstruction_3D.util.util_loadlib as loadlib
from Reconstruction_2D.test import Reconstruction_2D
from Reconstruction_3D.imageprocessing.silhoueete import silhouette_extraction
from Reconstruction_3D.util.util_print import str_error, str_stage, str_verbose
import warnings
warnings.filterwarnings("ignore")

test_sketch_path = r'/home/oem/Downloads/code/Reconstruction_2D/data/sketch/100stEpoch_515stBatch_23stRealA.png'
netG_weights_path = r'/home/oem/Downloads/code/Reconstruction_2D/parameters/Plane/100st_epoch_G_net.pth'
output_dir = '/home/oem/Downloads/code/output'
thresh_silhouette = 0.95

def sketch2model(test_sketch_path, netG_weights_path, output_dir, thresh_silhouette):

    class Options:
        def __init__(self):
            self.net_file = '/home/oem/Downloads/code/Reconstruction_3D/downloads/models/full_model.pt'
            self.overwrite = True

            self.sketch_dir = r''
            self.true_image_dir = r''
            self.category = '02691156'

            self.continue_train = True
            self.starting_epoch = 117  # from 1

            self.training_information_dir = r''
            self.save_input_sketches_folder_name = r''
            self.save_input_true_images_folder_name = r''
            self.save_output_fake_images_folder_name = r''
            self.save_training_log_folder_name = r''
            self.save_model_folder_name = r''

            # self.transform_size = 256
            self.image_size = 480
            self.aug_prob = 0.5
            self.NetInput_size = 256
            self.aug_fill = 'White'

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.gpu_ids = [self.device]
            # self.gpu_ids = []
            self.n_epochs = 300
            self.epoch_count = 1
            self.n_epochs_decay = 100
            self.direction = 'AtoB'
            self.isTrain = True
            self.input_nc = 3
            self.output_nc = 3
            self.ngf = 64
            self.ndf = 64
            self.netG = 'unet_256'
            self.netD = 'basic'
            self.n_layers_D = 3
            self.norm = 'instance'
            self.no_dropout = True
            self.init_type = 'normal'
            self.init_gain = 0.02
            self.gan_mode = 'vanilla'
            self.lr = 0.0002
            self.beta1 = 0.5
            self.lambda_L1 = 100.0
            self.lr_policy = 'linear'  # [linear | step | plateau | cosine]

            self.train_ratio = 0.8
            self.batch_size_2D = 40

            self.gpu = '0'
            self.manual_seed = 42
            self.resume = 0
            self.suffix = ''
            self.epoch = 10
            self.dataset = None
            self.workers = 4
            self.classes = 'car'
            self.batch_size = 1
            self.epoch_batches = None
            self.eval_batches = 1
            self.eval_at_start = True
            self.log_time = True
            self.net = 'genre_full_model'
            self.optim = 'adam'
            self.lr = 1e-4
            self.adam_beta1 = 0.5
            self.adam_beta2 = 0.9
            self.sgd_momentum = 0.9
            self.sgd_dampening = 0
            self.wdecay = 0.0
            self.logdir = '/home/oem/Downloads/code/Reconstruction_3D/logs'
            self.log_batch = True
            self.expr_id = 0
            self.save_net = 0
            self.save_net_opt = False
            self.vis_every_vali = 1
            self.vis_every_train = 1
            self.vis_batches_vali = 10
            self.vis_batches_train = 10
            self.tensorboard = True
            self.vis_workers = 4
            self.vis_param_f = None
            self.pred_depth_minmax = True
            self.joint_train = False
            self.load_offline = True
            self.inpaint_path = '/home/oem/Downloads/code/Reconstruction_3D/downloads/models/depth_pred_with_inpaint.pt'
            self.net1_path = '/home/oem/Downloads/code/Reconstruction_3D/downloads/models/depth_pred_with_inpaint.pt'
            self.padding_margin = 16
            self.surface_weight = 1.0
            self.canon_sup = True
            self.marrnet1 = '/home/oem/Downloads/code/Reconstruction_3D/downloads/models/marrnet1_with_minmax.pt'
            self.canon_voxel = True
            self.wgangp_lambda = float(10)
            self.wgangp_norm = float(1)
            self.gan_d_iter = 1

    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    sub_output_folder_name = f"output_{time_str}"
    sub_output_folder_path = os.path.join(output_dir, sub_output_folder_name)
    os.makedirs(sub_output_folder_path)

    opt = Options()

    fake_B = Reconstruction_2D(opt, test_sketch_path, netG_weights_path)
    fake_B_numpy = fake_B[0].permute(1, 2, 0).numpy()
    fake_B_resize = cv2.resize(fake_B_numpy, (opt.image_size, opt.image_size))
    fake_B_rescaled = (fake_B_resize * 255).astype(np.uint8)

    generated_rgb_path = os.path.join(sub_output_folder_path, "generated_rgb.png")
    cv2.imwrite(generated_rgb_path, fake_B_rescaled)

    fake_B_contour = silhouette_extraction(fake_B_numpy, opt.image_size, thresh_silhouette)
    # print(type(fake_B_contour))

    silhouette_path = os.path.join(sub_output_folder_path, "silhouette.png")
    cv2.imwrite(silhouette_path, fake_B_contour)


    opt.input_rgb = generated_rgb_path
    opt.input_mask = silhouette_path
    opt.output_dir = sub_output_folder_path



    print("Testing Pipeline")

    ###################################################

    print(str_stage, "Parsing arguments")
    # opt = options_test.parse()
    opt.full_logdir = None
    # print(opt)

    ###################################################

    print(str_stage, "Setting device")
    if opt.gpu == '-1':
        device = torch.device('cpu')
    else:
        loadlib.set_gpu(opt.gpu)
        device = torch.device('cuda')
    if opt.manual_seed is not None:
        loadlib.set_manual_seed(opt.manual_seed)

    ###################################################

    print(str_stage, "Setting up loggers")
    logger_list = [
        loggers.TerminateOnNaN(),
    ]
    logger = loggers.ComposeLogger(logger_list)

    ###################################################

    print(str_stage, "Setting up models")
    Model = models.get_model(opt.net, test=True)
    model = Model(opt, logger)
    model.to(device)
    model.eval()
    print(model)
    print("# model parameters: {:,d}".format(model.num_parameters()))

    ###################################################

    print(str_stage, "Setting up data loaders")
    start_time = time.time()
    Dataset = datasets.get_dataset('test')
    dataset = Dataset(opt, model=model)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    n_batches = len(dataloader)
    dataiter = iter(dataloader)
    print(str_verbose, "Time spent in data IO initialization: %.2fs" %
          (time.time() - start_time))
    print(str_verbose, "# test points: " + str(len(dataset)))
    print(str_verbose, "# test batches: " + str(n_batches))

    ###################################################

    print(str_stage, "Testing")
    for i in tqdm(range(n_batches)):
        batch = next(dataiter)
        model.test_on_batch(i, batch)

    sub_output_batch_path = os.path.join(sub_output_folder_path, 'batch0000')

    files_to_copy = os.listdir(sub_output_batch_path)

    for file_name in files_to_copy:
        if 'rgb' not in file_name:
            source_file_path = os.path.join(sub_output_batch_path, file_name)
            destination_file_path = os.path.join(sub_output_folder_path, file_name)
            shutil.copy(source_file_path, destination_file_path)

    print("over")

    return sub_output_folder_path

if __name__ == "__main__":
    sketch2model(test_sketch_path, netG_weights_path, output_dir, thresh_silhouette)