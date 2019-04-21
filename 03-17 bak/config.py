import time


class Config:
    lfw_data_dir = '/home/kevin/文档/LFW'
    celeba_data_dir = '/media/kevin/文档/CelebA200'
    use_funneled = False
    face_data = 'lfw'
    if face_data == 'lfw':
        imgset_data_dir = lfw_data_dir
        if use_funneled:
            img_data_dir = lfw_data_dir + '/lfw-deepfunneled/'
        else:
            img_data_dir = lfw_data_dir + '/lfw/'
        input_img_size = 250
        crop_size = 200  # 0 为不操作
        resize_size = 0  # 0 为不操作
    if face_data == 'celeba':
        imgset_data_dir = celeba_data_dir
        img_data_dir = celeba_data_dir + '/img_align_celeba_200/'
        input_img_size = 200
        crop_size = 0  # 0 为不操作
        resize_size = 0  # 0 为不操作
    log_dir = 'checkpoints/'
    train_net_name = 'face'
    init_lr = 0.001
    margin = 0.3
    train_bs = 16
    valid_bs = 16
    test_bs = 16
    max_epoch = 30
    show_plot_epoch = False
    train_harder = True

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
    lfw_data_dir = '/DATACENTER5/caoke/LFW'
    celeba_data_dir = '/DATACENTER5/caoke/CelebA200'
    use_funneled = False
    face_data = 'celeba'
    if face_data == 'lfw':
        imgset_data_dir = lfw_data_dir
        if use_funneled:
            img_data_dir = lfw_data_dir + '/lfw-deepfunneled/'
        else:
            img_data_dir = lfw_data_dir + '/lfw/'
        input_img_size = 250
        crop_size = 224  # 0 为不操作
        resize_size = 0  # 0 为不操作
    if face_data == 'celeba':
        imgset_data_dir = celeba_data_dir
        img_data_dir = celeba_data_dir + '/img_align_celeba_200/'
        input_img_size = 200
        crop_size = 0  # 0 为不操作
        resize_size = 224  # 0 为不操作
    log_dir = 'checkpoints/'
    train_net_name = 'face'
    init_lr = 0.001
    margin = 0.5
    train_bs = 64
    valid_bs = 12
    test_bs = 12
    max_epoch = 30
    show_plot_epoch = False
    train_harder = True

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
