import torch
from Reconstruction_2D import model
import matplotlib.pyplot as plt
import warnings
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings("ignore")


class Options:
    def __init__(self):
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


def Reconstruction_2D(opt, test_sketch_path, netG_weights_path):

    training_model = model.Pix2PixModel(opt)

    training_model.netG.load_state_dict(torch.load(netG_weights_path))

    transform = transforms.Compose([
        transforms.Resize((opt.NetInput_size, opt.NetInput_size)),  # 调整图像大小
        transforms.ToTensor(),           # 转换为张量
    ])

    sketch = Image.open(test_sketch_path)

    if 'A' in sketch.getbands() or 'L' in sketch.getbands():
        sketch = sketch.convert('RGB')

    real_A = transform(sketch)
    real_A = real_A.unsqueeze(0)
    real_A = real_A.cuda()


    fake_B = training_model.netG(real_A)
    fake_B = fake_B.cpu()
    fake_B = fake_B.detach()

    return fake_B

if __name__ == '__main__':
    opt = Options()
    test_sketch_path = r'/home/oem/Downloads/code/Reconstruction_2D/data/sketch/100stEpoch_515stBatch_40stRealA.png'
    netG_weights_path = r'/home/oem/Downloads/code/Reconstruction_2D/parameters/100st_epoch_G_net.pth'

    fake_B = Reconstruction_2D(opt, test_sketch_path, netG_weights_path)

    plt.imshow(fake_B[0].permute(1, 2, 0))
    plt.title('result')
    plt.axis('off')

    # 显示图片
    plt.show()