import torch
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message="Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).")


class PairedImageAugmentation:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, real_A, real_B):
        print('Data Augmentation:')
        print()

        batch_size_A, c_A, h_A, w_A = real_A.shape
        batch_size_B, c_B, h_B, w_B = real_B.shape

        if w_A != w_B or h_A != h_B:
            assert False, "Please transform real_A and real_B to the same size, if you want to execute data augmentation"

        if self.opt.aug_fill != 'White':
            assert False, "Actual program can only be filled by white"

        # # Convert the images back to PIL format
        # real_A = transforms.ToPILImage()(real_A)
        # real_B = transforms.ToPILImage()(real_B)

        # Convert the images back to PIL format
        real_A_images = []
        real_B_images = []
        for image in range(batch_size_A):
            real_A_images.append(transforms.ToPILImage()(real_A[image]))
            real_B_images.append(transforms.ToPILImage()(real_B[image]))

        # Random flipping
        for i_flipping in range(batch_size_A):
            # if 2 < self.opt.aug_prob:
            if random.random() < self.opt.aug_prob:
                if random.random() < 0.5:
                    # real_A = F.hflip(real_A)
                    # real_B = F.hflip(real_B)
                    real_A_images[i_flipping] = F.hflip(real_A_images[i_flipping])
                    real_B_images[i_flipping] = F.hflip(real_B_images[i_flipping])
                else:
                    real_A = F.vflip(real_A)
                    real_B = F.vflip(real_B)
                    real_A_images[i_flipping] = F.vflip(real_A_images[i_flipping])
                    real_B_images[i_flipping] = F.vflip(real_B_images[i_flipping])

        # Random rotation
        for i_rotation in range(batch_size_A):
            if random.random() < self.opt.aug_prob:
            # if 2 < self.opt.aug_prob:
                angle = random.uniform(-180, 180)

                # real_A = F.rotate(real_A, angle, fill=[255, 255, 255])
                # real_B = F.rotate(real_B, angle, fill=[255, 255, 255])

                real_A_images[i_rotation] = F.rotate(real_A_images[i_rotation], angle, fill=[255, 255, 255])
                real_B_images[i_rotation] = F.rotate(real_B_images[i_rotation], angle, fill=[255, 255, 255])


        # Random scaling
        for i_scaling in range(batch_size_A):
            # if 2 < self.opt.aug_prob:
            if random.random() < self.opt.aug_prob:
                scaling_factor = random.uniform(0.75, 1.25)
                new_size_A = [int(x * scaling_factor) for x in real_A_images[i_scaling].size[::-1]]
                new_size_B = [int(x * scaling_factor) for x in real_B_images[i_scaling].size[::-1]]
                # real_A = F.resize(real_A, new_size_A, interpolation=F.InterpolationMode.BILINEAR)
                # real_B = F.resize(real_B, new_size_B, interpolation=F.InterpolationMode.BILINEAR)
                real_A_images[i_scaling] = F.resize(real_A_images[i_scaling], new_size_A, interpolation=F.InterpolationMode.BILINEAR)
                real_B_images[i_scaling] = F.resize(real_B_images[i_scaling], new_size_A, interpolation=F.InterpolationMode.BILINEAR)

                # Create a blank white image with the desired size (256x256)
                background_image_A = Image.new('RGB', (self.opt.input_size, self.opt.input_size), 'white')
                background_image_B = Image.new('RGB', (self.opt.input_size, self.opt.input_size), 'white')

                # Calculate the center of the rotated image
                w, h = real_A_images[i_scaling].size
                center = (w // 2, h // 2)

                # Paste the rotated image onto the blank image, using the center as the anchor point
                background_image_A.paste(real_A_images[i_scaling], (center[0] - w // 2, center[1] - h // 2))
                background_image_B.paste(real_B_images[i_scaling], (center[0] - w // 2, center[1] - h // 2))

                # Update real_A and real_B with the new images
                real_A_images[i_scaling] = background_image_A
                real_B_images[i_scaling] = background_image_B


        # Random translation
        for i_translation in range(batch_size_A):
            if random.random() < self.opt.aug_prob:
            # if 0 < self.opt.aug_prob:

                max_translate = int(self.opt.input_size * 0.3)

                # generate the distance of translation
                x_translate = random.randint(-max_translate, max_translate)
                y_translate = random.randint(-max_translate, max_translate)

                # generate a big white background
                expanded_A = Image.new('RGB', (w_A + 2 * max_translate, h_A + 2 * max_translate), (255, 255, 255))
                expanded_B = Image.new('RGB', (w_B + 2 * max_translate, h_B + 2 * max_translate), (255, 255, 255))

                # paste image on the background and move it
                expanded_A.paste(real_A_images[i_translation], (max_translate + x_translate, max_translate + y_translate))
                expanded_B.paste(real_B_images[i_translation], (max_translate + x_translate, max_translate + y_translate))

                # crop to original dimension
                real_A_images[i_translation] = expanded_A.crop((max_translate, max_translate, max_translate + w_A, max_translate + h_A))
                real_B_images[i_translation] = expanded_B.crop((max_translate, max_translate, max_translate + w_B, max_translate + h_B))


        # Randon color
        for i_color in range(batch_size_A):
            if random.random() < self.opt.aug_prob:
            # if 2 < self.opt.aug_prob:
                # Define the color jitter transformation
                color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
                real_A_images[i_color] = color_jitter(real_A_images[i_color])
                real_B_images[i_color] = color_jitter(real_B_images[i_color])

        # Transform image to Tensor format
        transform_to_tensor = transforms.Compose([transforms.ToTensor()])

        real_A = torch.stack([transform_to_tensor(image) for image in real_A_images], dim=0)
        real_B = torch.stack([transform_to_tensor(image) for image in real_B_images], dim=0)
        # real_B = torch.stack(real_B_images, dim=0)

        # Transform : resize and normalization
        transform_list = []
        transform_list.append(transforms.Resize(self.opt.input_size))
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform_of_augmentation = transforms.Compose(transform_list)

        for i_transform in range(batch_size_A):
            real_A[i_transform] = transform_of_augmentation(real_A[i_transform])
            real_B[i_transform] = transform_of_augmentation(real_B[i_transform])

        return real_A, real_B
# ----------------------------------------------------------------------------------------------------------------------
# # Convert the images back to PIL format
            # real_A = transforms.ToPILImage()(real_A)
            # real_B = transforms.ToPILImage()(real_B)
# ------

class PairedImageDataset(data.Dataset):
    def __init__(self, sketches_filenames, true_images_filenames, sketches_dir, true_images_dir):
        self.sketches_filenames = sketches_filenames
        self.true_images_filenames = true_images_filenames
        self.sketches_dir = sketches_dir
        self.true_images_dir = true_images_dir
        self.transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        sketch_filename = self.sketches_filenames[index]
        true_image_filename = self.true_images_filenames[index]
        sketch_path = os.path.join(self.sketches_dir, sketch_filename)
        true_image_path = os.path.join(self.true_images_dir, true_image_filename)

        # 读取草图和真实图片
        sketch = Image.open(sketch_path).convert('RGB')
        sketch = self.transform_to_tensor(sketch)
        true_image = Image.open(true_image_path).convert('RGB')
        true_image = self.transform_to_tensor(true_image)

        return sketch, true_image

    def __len__(self):
        return len(self.sketches_filenames)
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    class Options:
        def __init__(self):
            self.sketch_dir = r'\\CAEN-NAS\commun\_METIERS\R&I\45_CONVERSION SCHEMAS\bdd\pokemon_pix2pix_dataset\demo\sketch_image'
            self.true_image_dir = r'\\CAEN-NAS\commun\_METIERS\R&I\45_CONVERSION SCHEMAS\bdd\pokemon_pix2pix_dataset\demo\true_image'
            self.train_ratio = 0.8
            self.batch_size = 2
            self.aug_fill = 'White'
            self.aug_prob = 0.5
            self.input_size = 256



    def split_dataset(sketches_filenames, true_images_filenames, test_size):
        train_sketches_filenames, test_sketches_filenames, train_true_images_filenames, test_true_images_filenames = train_test_split(
            sketches_filenames, true_images_filenames, test_size=test_size,
            random_state=None)  # recommended random state = 42

        print(type(train_sketches_filenames))

        train_dataset = PairedImageDataset(train_sketches_filenames, train_true_images_filenames,
                                                         opt.sketch_dir,
                                                         opt.true_image_dir)
        test_dataset = PairedImageDataset(test_sketches_filenames, test_true_images_filenames,
                                                        opt.sketch_dir,
                                                        opt.true_image_dir)

        return train_dataset, test_dataset



    opt = Options()

    sketches_filenames = sorted(os.listdir(opt.sketch_dir))
    true_images_filenames = sorted(os.listdir(opt.true_image_dir))

    # dataset = PairedImageDataset(sketches_filenames, true_images_filenames, opt.sketch_dir,
    #                                            opt.true_image_dir)

    train_dataset, test_dataset = split_dataset(sketches_filenames, true_images_filenames,
                                                test_size=1 - opt.train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    DataAugmentation = PairedImageAugmentation(opt)

    for epoch in range(1):
        for batch_idx, (sketches, true_images) in enumerate(train_loader):
            print(f'type of sketches : {type(sketches)}')
            print(f'shape of sketches : {sketches.shape}')
            print(f'type of true images : {type(true_images)}')
            print(f'shape of true images : {true_images.shape}')
            print()

            real_A, real_B = DataAugmentation(sketches, true_images)
            print(f'type of real_A : {type(real_A)}')
            print(f'shape of real_A : {real_A.shape}')
            print(f'type of real_B : {type(real_B)}')
            print(f'shape of real_B : {real_B.shape}')
            print()


            for i in range(opt.batch_size):
                # 显示草图图片
                plt.subplot(1, 2, 1)
                plt.imshow(real_A[i].permute(1, 2, 0))
                plt.title('Sketch Image')
                plt.axis('off')

                # 显示真实图片
                plt.subplot(1, 2, 2)
                plt.imshow(real_B[i].permute(1, 2, 0))
                plt.title('True Image')
                plt.axis('off')

                # 显示图片
                plt.show()

            break
