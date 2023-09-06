# from models import create_model
# from util.visualizer import Visualizer
import torch
import model
import os
import new_preprocessing
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from datetime import datetime
import warnings
import time




'''
note : 
1. Square pictures would be better
'''


class Options:
    def __init__(self):
        self.sketch_dir = r'D:\stage\bdd\2dReconstruction\02691156\sketch_image'
        self.true_image_dir = r'D:\stage\bdd\2dReconstruction\02691156\true_image'
        self.category = '02691156'

        self.continue_train = True
        self.netD_weights_path = r'D:\stage\code\V7\informations of training\training_2023-08-03-10-05-22\6_model\116st_epoch_D_net.pth'
        self.netG_weights_path = r'D:\stage\code\V7\informations of training\training_2023-08-03-10-05-22\6_model\116st_epoch_G_net.pth'
        self.starting_epoch = 117  # from 1

        self.training_information_dir = r'D:\stage\code\V7\informations of training'
        self.save_input_sketches_folder_name = r'2_input images\1_sketches'
        self.save_input_true_images_folder_name = r'2_input images\2_true_images'
        self.save_output_fake_images_folder_name = r'3_output images'
        self.save_training_log_folder_name = r'4_training log'
        self.save_model_folder_name = r'6_model'

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
        self.batch_size = 40


def split_dataset(items, test_size, opt):
    train_items, test_items = train_test_split(items, test_size=test_size, random_state=None)  # recommended random state = 42

    train_sketches_filenames = new_preprocessing.CreateFilenanmes(train_items, 'sketch', opt)
    train_true_images_filenames = new_preprocessing.CreateFilenanmes(train_items, 'true_image', opt)
    test_sketches_filenames = new_preprocessing.CreateFilenanmes(test_items, 'sketch', opt)
    test_true_images_filenames = new_preprocessing.CreateFilenanmes(test_items, 'true_image', opt)

    train_dataset = new_preprocessing.PairedImageDataset(train_sketches_filenames, train_true_images_filenames, opt.sketch_dir, opt.true_image_dir)
    test_dataset = new_preprocessing.PairedImageDataset(test_sketches_filenames, test_true_images_filenames,
                                      opt.sketch_dir,
                                      opt.true_image_dir)

    return train_items, test_items, train_dataset, test_dataset


if __name__ == '__main__':
    '''set all hyper parameters'''
    opt = Options()

    # set root dir
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    ThisTrainingInfoSave = os.path.join(opt.training_information_dir, 'training_' + current_time)

    # set all folders
    training_log_file = f'training_log_{current_time}.txt'
    training_log_folder = os.path.join(ThisTrainingInfoSave, opt.save_training_log_folder_name)
    real_A_folder = os.path.join(ThisTrainingInfoSave, opt.save_input_sketches_folder_name)
    real_B_folder = os.path.join(ThisTrainingInfoSave, opt.save_input_true_images_folder_name)
    fake_B_folder = os.path.join(ThisTrainingInfoSave, opt.save_output_fake_images_folder_name)
    model_folder = os.path.join(ThisTrainingInfoSave, opt.save_model_folder_name)
    os.makedirs(training_log_folder, exist_ok=True)
    os.makedirs(real_A_folder, exist_ok=True)
    os.makedirs(real_B_folder, exist_ok=True)
    os.makedirs(fake_B_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)



    with open(os.path.join(training_log_folder, training_log_file), 'w') as file_training_log:
        items = sorted(os.listdir(opt.sketch_dir))

        '''set train set and test set'''
        train_items, test_items, train_dataset, test_dataset = split_dataset(items, 1 - opt.train_ratio, opt)
        print(train_dataset)
        print(test_dataset)
        file_training_log.write(f'    length of train set : {len(train_dataset)}\n')
        file_training_log.write(f'    length of test set : {len(test_dataset)}\n')
        file_training_log.write(f'    items for training:\n')
        for item1 in train_items:
            file_training_log.write(f'{item1}\n')
        file_training_log.write(f'    items for test:\n')
        for item2 in test_items:
            file_training_log.write(f'{item2}\n')
        file_training_log.write('\n')

        '''set batches'''
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

        '''set data augmentation'''
        DataAugmentation = new_preprocessing.PairedImageAugmentation(opt)

        '''set model and initialize it'''
        training_model = model.Pix2PixModel(opt)

        '''continue previous training if necessary'''
        if opt.continue_train is True:
            training_model.netD.load_state_dict(torch.load(opt.netD_weights_path))
            training_model.netG.load_state_dict(torch.load(opt.netG_weights_path))

        print("training :")
        for epoch in range(opt.starting_epoch-1, opt.n_epochs):
            start_epoch_time = time.time()
            total_loss_D = 0
            total_loss_G = 0
            file_training_log.write(f'epoch : {epoch + 1}\n')
            print(f'epoch : {epoch + 1}')
            print()

            '''update learning rates'''
            training_model.update_learning_rate()

            # i = 0
            for batch_idx, (sketches, true_images) in enumerate(train_loader):
                start_batch_time = time.time()
                file_training_log.write(f'    batch : {batch_idx + 1}\n')

                '''data augmentation'''
                real_A, real_B = DataAugmentation(sketches, true_images)
                real_A, real_B = real_A.cuda(), real_B.cuda()

                '''set input and import it in the model'''
                input = {
                    'A': real_A,
                    'A_paths': 'xxx',
                    'B': real_B,
                    'B_paths': 'xxx'
                }

                training_model.set_input(input)

                '''generate fake_B'''
                fake_B = training_model.netG(real_A)
                fake_B = fake_B.cuda()

                '''
                Optimize :
                1. compute fake images: G(A)
                2. update D
                    2.1 enable backprop for D
                    2.2 set D's gradients to zero
                    2.3 calculate gradients for D
                    2.4 update D's weights
                3. update G
                    3.1 D requires no gradients when optimizing G
                    3.2 set G's gradients to zero
                    3.3 calculate graidents for G
                    3.4 update G's weights
                '''
                training_model.optimize_parameters()

                '''calculate loss of this batch and save it'''
                loss_D = training_model.loss_D
                loss_G = training_model.loss_G
                print(f"loss D of epoch{epoch+1} batch{batch_idx+1} is {loss_D}")
                print(f"loss G of epoch{epoch+1} batch{batch_idx+1} is {loss_G}")
                file_training_log.write(f'    loss_D : {loss_D}\n')
                file_training_log.write(f'    loss_G : {loss_G}\n')

                total_loss_D += loss_D
                total_loss_G += loss_G

                end_batch_time = time.time()
                record_batch_time= end_batch_time - start_batch_time
                file_training_log.write(f'    time used : {record_batch_time}\n')
                file_training_log.write('\n')
                print(f"time used in the epoch{epoch+1} batch{batch_idx+1} is {record_batch_time}s")

            '''save input images in the training information folder'''
            real_A, real_B = real_A.cpu(), real_B.cpu()
            for i_real in range(opt.batch_size):
                filename_real_A = f"{epoch + 1}stEpoch_{batch_idx + 1}stBatch_{i_real + 1}stRealA.png"
                filename_real_B = f"{epoch + 1}stEpoch_{batch_idx + 1}stBatch_{i_real + 1}stRealB.png"

                save_path_RealA = os.path.join(real_A_folder, filename_real_A)
                save_path_RealB = os.path.join(real_B_folder, filename_real_B)

                save_image(real_A[i_real], save_path_RealA)
                save_image(real_B[i_real], save_path_RealB)

            '''save output images in the training information folder'''
            fake_B = fake_B.cpu()
            for i_fake in range(opt.batch_size):
                filename_fake_B = f"{epoch + 1}stEpoch_{batch_idx + 1}stBatch_{i_fake + 1}stFakeB.png"
                save_path_FakeB = os.path.join(fake_B_folder, filename_fake_B)
                save_image(fake_B[i_fake], save_path_FakeB)

            '''save model'''
            for model_name in training_model.model_names:
                if isinstance(model_name, str):
                    # filename_model = '%s_net_%s.pth' % (epoch, model_name)
                    filename_model = f'{epoch+1}st_epoch_{model_name}_net.pth'
                    save_path_model = os.path.join(model_folder, filename_model)
                    net = getattr(training_model, 'net' + model_name)

                    torch.save(net.cpu().state_dict(), save_path_model)
                    net.cuda()

            # record learning rate of every epoch
            learning_rate = training_model.optimizers[0].param_groups[0]['lr']
            file_training_log.write(f'learning rate of {epoch + 1}st epoch is : {learning_rate}\n')
            print(f"The learning rate of epoch{epoch+1} is {learning_rate}")

            # record average loss of every epoch
            loss_D_epoch = total_loss_D / batch_idx
            loss_G_epoch = total_loss_G / batch_idx
            file_training_log.write(f'loss_D of {epoch + 1}st epoch is : {loss_D_epoch}\n')
            file_training_log.write(f'loss_G of {epoch + 1}st epoch is : {loss_G_epoch}\n')
            print(f"Average loss D of epoch{epoch+1} is {loss_D_epoch}")
            print(f"Average loss G of epoch{epoch+1} is {loss_G_epoch}")

            end_epoch_time = time.time()
            record_epoch_time = end_epoch_time - start_epoch_time
            file_training_log.write(f'time used of {epoch + 1}st epoch is : {record_epoch_time}s\n')
            file_training_log.write('\n')
            print(f"time used in the epoch{epoch+1} is {record_epoch_time}s")
            print()

    file_training_log.close()
