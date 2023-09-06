import os
import time
from shutil import rmtree
from tqdm import tqdm
import torch
from options import options_test
import datasets
import models
from util.util_print import str_error, str_stage, str_verbose
import util.util_loadlib as loadlib
from loggers import loggers
from imageprocessing.silhoueete import silhouette_extraction

input_iamge = '/home/oem/Downloads/code/Reconstruction_3D/downloads/data/test/genre/300stEpoch_515stBatch_7stRealBresized.png'
output_dir = '/Reconstruction_3D/output'

class Options:
    def __init__(self, input_iamge, output_dir):
        self.input_rgb = input_iamge
        self.input_mask = '/home/oem/Downloads/code/Reconstruction_3D/downloads/data/test/genre/300stEpoch_515stBatch_7stRealBsilhouette.png'
        self.net_file = '/Reconstruction_3D/downloads/models/full_model.pt'
        self.output_dir = output_dir
        self.overwrite = True

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
        self.inpaint_path = '/Reconstruction_3D/downloads/models/depth_pred_with_inpaint.pt'
        self.net1_path = '/Reconstruction_3D/downloads/models/depth_pred_with_inpaint.pt'
        self.padding_margin = 16
        self.surface_weight = 1.0
        self.canon_sup = True
        self.marrnet1 = '/home/oem/Downloads/code/Reconstruction_3D/downloads/models/marrnet1_with_minmax.pt'
        self.canon_voxel = True
        self.wgangp_lambda = float(10)
        self.wgangp_norm = float(1)
        self.gan_d_iter = 1


opt = Options(input_iamge, output_dir)

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

# print(str_stage, "Setting up output directory")
# output_dir = opt.output_dir
# output_dir += ('_' + opt.suffix.format(**vars(opt))) \
#     if opt.suffix != '' else ''
# opt.output_dir = output_dir
#
# if os.path.isdir(output_dir):
#     if opt.overwrite:
#         rmtree(output_dir)
#     else:
#         raise ValueError(str_error +
#                          " %s already exists, but no overwrite flag"
#                          % output_dir)
# os.makedirs(output_dir)

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

print("over")