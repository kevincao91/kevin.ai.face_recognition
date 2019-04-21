import time


class Config:
    lfw_data_dir = '/media/kevin/文档/DataSet/LFW'
    celeba_data_dir = '/media/kevin/文档/CelebA200'
    imgset_data_dir = './dataset'
    test_data_dir = './dataset'
    use_funneled = False
    face_data = 'lfw'
    if face_data == 'lfw':
        n_people = 5749
        if use_funneled:
            img_data_dir = lfw_data_dir + '/lfw-deepfunneled/'
        else:
            img_data_dir = lfw_data_dir + '/lfw/'
        input_img_size = 250
        crop_size = 224  # 0 为不操作
        resize_size = 0  # 0 为不操作
    if face_data == 'celeba':
        n_people = 10177
        img_data_dir = celeba_data_dir + '/img_align_celeba_200/'
        input_img_size = 200
        crop_size = 0  # 0 为不操作
        resize_size = 224  # 0 为不操作
    train_net_name = 'resnet18'
    input_net_size = 0
    init_lr = 0.0001
    margin = 0.5
    train_bs = 3
    valid_bs = 3
    test_bs = 3
    max_epoch = 3
    show_plot_epoch = False
    train_harder = True
    log_dir = None  # 程序中定义

    def __init__(self):
        self._get_input_net_size()

    def _get_input_net_size(self):
        if self.crop_size == 0 and self.resize_size == 0:
            self.input_net_size = self.input_img_size
        elif self.resize_size == 0:
            self.input_net_size = self.crop_size
        else:
            self.input_net_size = self.resize_size

    def _print_opt(self):
        time_str = time.strftime('%m%d%H%M%S')
        config_txt_path = self.log_dir + time_str + '_config.txt'
        with open(config_txt_path, 'w') as f:
            state_dict = self._state_dict()
            string = '===================user config==================='
            print(string)
            f.write(string + '\n')
            for k, v in state_dict.items():
                string = str(k) + '\t\t\t' + str(v)
                print(string)
                f.write(string + '\n')
            string = '=======================end======================='
            print(string)
            f.write(string + '\n')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()


class CUDAConfig:
    lfw_data_dir = '/media/kevin/文档/DataSet/LFW'
    celeba_data_dir = '/DATACENTER5/caoke/CelebA200'
    imgset_data_dir = './dataset'
    test_data_dir = './dataset'
    use_funneled = False
    face_data = 'lfw'
    if face_data == 'lfw':
        n_people = 5749
        if use_funneled:
            img_data_dir = lfw_data_dir + '/lfw-deepfunneled/'
        else:
            img_data_dir = lfw_data_dir + '/lfw/'
        input_img_size = 250
        crop_size = 224  # 0 为不操作
        resize_size = 0  # 0 为不操作
    if face_data == 'celeba':
        n_people = 10177
        img_data_dir = celeba_data_dir + '/img_align_celeba_200/'
        input_img_size = 200
        crop_size = 0  # 0 为不操作
        resize_size = 224  # 0 为不操作
    log_dir = 'checkpoints/'
    train_net_name = 'facenet'
    input_net_size = 0
    init_lr = 0.0001
    alpha = 10
    margin = 0.5 * alpha
    train_bs = 24
    valid_bs = 12
    test_bs = 12
    max_epoch = 100
    show_plot_epoch = False
    train_harder = True
    log_dir = None  # 程序中定义

    def __init__(self):
        self._get_input_net_size()

    def _get_input_net_size(self):
        if self.crop_size == 0 and self.resize_size == 0:
            self.input_net_size = self.input_img_size
        elif self.resize_size == 0:
            self.input_net_size = self.crop_size
        else:
            self.input_net_size = self.resize_size

    def _print_opt(self):
        time_str = time.strftime('%m%d%H%M%S')
        config_txt_path = self.log_dir + time_str + '_config.txt'
        with open(config_txt_path, 'w') as f:
            state_dict = self._state_dict()
            string = '===================user config==================='
            print(string)
            f.write(string + '\n')
            for k, v in state_dict.items():
                string = str(k) + '\t\t\t' + str(v)
                print(string)
                f.write(string + '\n')
            string = '=======================end======================='
            print(string)
            f.write(string + '\n')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in CUDAConfig.__dict__.items()
                if not k.startswith('_')}


cuda_opt = CUDAConfig()
