import os
import glob
import imageio
from revide_dataset import data_utils as utils
import torch.utils.data as data


class RevideDataset(data.Dataset):
    """_summary_
    This is a torch dataset that pulls in REVIDE data. More info can be found in the README.
    Example usage:
    
    return DataLoader(
        dataset=RevideDataset(....args...),
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=not is_cpu,
        num_workers=number_of_threads
    )
    
    """
    def __init__(self, 
        train_directory: str,
        test_directory: str,
        name: str="REVIDE", 
        patch_size: int=256,
        load_all_on_ram: bool=False,
        test_every_x_batch: int=1000,
        batch_size: int=8,
        no_data_augmentation: bool=False,
        size_must_mode: int=1,
        max_rgb_value: int=3,
        train=True, 
        clear_folder: str='GT', 
        hazy_folder: str='INPUT'):
        """_summary_

        Args:
            train_directory (str): The directory to the train folder.
            test_directory (str): The directory to the test folder.
            name (str): The nae of the dataset. REVIDE is probably a good choice.
            patch_size (int): Output patch size.
            load_all_on_ram (bool): Should we load all datasets at once on RAM?
            test_every_x_batch (int): Test per every N batches
            batch_size (int): Input batch size for training
            no_data_augmentation (bool): Do not use data augmentation.
            size_must_mode (int): The size of the network input must mode this number.
            max_rgb_value (int): Tbhe maximum value of RGB.
            train (bool, optional): Is this for training? Otherwise, we assume it is testing. Defaults to True.
            clear_folder (str, optional): The name of the clear/ground truth folder that holds the photos. Defaults to 'GT'.
            hazy_folder (str, optional): The name of the folder that holds the hazy photos. Defaults to 'INPUT'.
        """
        
        self.name = name
        self.train = train
        self.clear_folder = clear_folder
        self.hazy_folder = hazy_folder
        self.patch_size = patch_size
        self.size_must_mode = size_must_mode
        self.max_rgb_value = max_rgb_value
        # Do you want to load all the data at once on RAM?
        self.load_all_on_ram = load_all_on_ram
        # Do we want to use data augmentation?
        self.no_data_augmentation = no_data_augmentation

        # Set the clear and hazy file systems
        # Choose a directory based on whether we are looking at the training dataset or the testing dataset
        if train:
            self._set_filesystem(train_directory)
        else:
            self._set_filesystem(test_directory)

        self.images_gt, self.images_input = self._scan()

        self.num_image = len(self.images_gt)
        print("Number of images to load:", self.num_image)

        if train:
            self.repeat = max(test_every_x_batch // max((self.num_image // batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

        if self.load_all_on_ram:
            self.data_gt, self.data_input = self._load(self.images_gt, self.images_input)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        # apath is just the path to the dataset. It should contain a "clear "images" folder and a 
        # "hazy images" folder, though they can be named different things depending on the child 
        # class' implementation
        self.apath = dir_data

        # Set the clear path by getting the full path to the dataset + adding the clear folder to the end
        self.dir_gt = os.path.join(self.apath, self.clear_folder)
        self.dir_input = os.path.join(self.apath, self.hazy_folder)
        print(f'DataSet clear/ground truth path:', self.dir_gt)
        print(f'DataSet hazy/input path:', self.dir_input)

    def _scan(self):
        """
        This scans in a specific way for the revide data, which is nested in video folders.
        For example:

        | video 1
            | frame 1 image of video 1
            | frame 2 image of video 1
        | video 2
            | frame 1 image of video 2
            | frame 2 image of video 2

        Returns:
            a tuple that has (list of paths for clear/ground truth images, list of paths for hazy/input images)
        """
        clear_image_paths: list[str] = []

        for folder_path in glob.glob(os.path.join(self.dir_gt, '*')):
            clear_image_paths.extend(glob.glob(os.path.join(folder_path, '*')))

        hazy_image_paths: list[str] = []
        for folder_path in glob.glob(os.path.join(self.dir_input, '*')):
            hazy_image_paths.extend(glob.glob(os.path.join(folder_path, '*')))

        assert len(clear_image_paths) == len(hazy_image_paths), f'The number of clear images must match the number of hazy images: clear {len(clear_image_paths)} hazy {len(hazy_image_paths)}'

        return sorted(clear_image_paths), sorted(hazy_image_paths)

    def _load(self, names_gt, names_input):
        print('Loading image dataset...')
        data_input = [imageio.imread(filename)[:, :, :3] for filename in names_input]
        data_gt = [imageio.imread(filename)[:, :, :3] for filename in names_gt]
        return data_gt, data_input

    def __getitem__(self, idx):
        """
        This is the main method that needs to be implemented for dataset, which supports fetching a data sample for 
        a given key.
        """
        if self.load_all_on_ram:
            input, gt, filename = self._load_file_from_loaded_data(idx)
        else:
            input, gt, filename = self._load_file(idx)

        input, gt = self.get_patch(input, gt, self.size_must_mode)
        input_tensor, gt_tensor = utils.np2Tensor(input, gt, rgb_range=self.max_rgb_value)

        return input_tensor, gt_tensor, filename

    def __len__(self):
        if self.train:
            return self.num_image * self.repeat
        else:
            return self.num_image

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_gt)
        else:
            return idx

    def _load_file(self, index):
        """
        Load the files.
        """
        index = self._get_index(index)
        # Get the paths for the ground truth/clear image and the hazy/input image
        f_gt = self.images_gt[index]
        f_input = self.images_input[index]

        assert os.path.basename(f_gt) == os.path.basename(f_input), f'The frames need to be the same: clear {f_gt} hazy {f_input}'

        # Since the file loading is more complicated, ensure they are coming from the same video
        ground_truth_video = os.path.basename(os.path.dirname(f_gt))
        input_video = os.path.basename(os.path.dirname(f_input))
        assert ground_truth_video == input_video, f'The videos need to be the same: clear {f_gt} hazy {f_input}'

        # read the images
        gt = imageio.imread(f_gt)[:, :, :3]
        input = imageio.imread(f_input)[:, :, :3]
        filename, _ = os.path.splitext(os.path.basename(f_gt))
        return input, gt, filename

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)
        gt = self.data_gt[idx]
        input = self.data_input[idx]
        filename = os.path.splitext(os.path.split(self.images_gt[idx])[-1])[0]
        return input, gt, filename

    def get_patch(self, input, gt, size_must_mode=1):
        if self.train:
            input, gt = utils.get_patch(input, gt, patch_size=self.patch_size)
            h, w, _ = input.shape
            if h != self.patch_size or w != self.patch_size:
                input = utils.bicubic_resize(input, size=(self.patch_size, self.patch_size))
                gt = utils.bicubic_resize(gt, size=(self.patch_size, self.patch_size))
                h, w, _ = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
            if not self.no_data_augmentation:
                input, gt = utils.data_augment(input, gt)
        else:
            h, w, _ = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
        return input, gt
