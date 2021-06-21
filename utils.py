import os
import torch
import time

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def save_checkpoint(state, folder_name, epoch=None):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    current = os.getcwd()
    path = os.path.join(current, 'checkpoint', folder_name)
    if epoch is not None:
        path = os.path.join(path, str(epoch))
    if not os.path.isdir(path):
        os.mkdir(path)
    os.chdir(path)
    torch.save(state, './ckpt.t7')
    os.chdir(current)

def load_checkpoint(folder_name, device, epoch=None):
    current = os.getcwd()
    if epoch is not None:
        os.chdir(os.path.join(current, 'checkpoint', folder_name, str(epoch)))
    else:
        os.chdir(os.path.join(current, 'checkpoint', folder_name))
    state = torch.load('./ckpt.t7', map_location=device)
    os.chdir(current)
    return state

def save_train_status(data, folder_name):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    current = os.getcwd()
    path = os.path.join(current, 'checkpoint', folder_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    os.chdir(path)
    with open('./train_status.txt', 'a') as f:
        for d in data:
            f.write(str(d)+' ')
        f.write('\n')
    os.chdir(current)

def save_test_status(data, folder_name):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    current = os.getcwd()
    path = os.path.join(current, 'checkpoint', folder_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    os.chdir(path)
    with open('./test_status.txt', 'a') as f:
        for d in data:
            f.write(str(d)+' ')
        f.write('\n')
    os.chdir(current)

class Timer():
    func_name = ['float2half','single2half','quantize','single2quarter','half2quarter','quarter2real','single2bfloat','input_quantize']
    func_num = len(func_name)
    func_time = [0]*func_num
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.function(*args, **kwargs)
        end_time = time.time()
        for i in range(Timer.func_num):
            if self.function.__name__ is Timer.func_name[i]:
                Timer.func_time[i] += end_time - start_time
                break
        return result

    @classmethod
    def show_time(cls, clear=False):
        for i in range(Timer.func_num):
            print("{} function's running time is {}s".format(Timer.func_name[i], Timer.func_time[i]))
            if clear:
                Timer.func_time[i] = 0

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1) if check_hash else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    if zipfile.is_zipfile(cached_file):
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            members = cached_zipfile.infolist()
            if len(members) != 1:
                raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
            cached_zipfile.extractall(model_dir)
            extraced_name = members[0].filename
            cached_file = os.path.join(model_dir, extraced_name)

    return torch.load(cached_file, map_location=map_location)

def check_nan_forward(self, input, output):
    if type(input) is tuple:
        for i in range(len(input)):
            if isnan(input[i].max()):
                print('Inside '+self.__class__.__name__+' forward pass input['+str(i+1)+'] occured nan!!', input[i].size())
    else:
        if isnan(input.max()):
            print('Inside '+self.__class__.__name__+' forward pass input occured nan!!', input.size())
    if type(output) is tuple:
        for i in range(len(output)):
            if isnan(output[i].max()):
                print('Inside '+self.__class__.__name__+' forward pass output['+str(i+1)+'] occured nan!!', output[i].size())
    else:
        if isnan(output.max()):
            print('Inside '+self.__class__.__name__+' forward pass output occured nan!!', output.size())

def check_nan_backward(self, grad_input, grad_output):
    if type(grad_input) is tuple:
        for i in range(len(grad_input)):
            if isnan(grad_input[i].max()):
                print('Inside '+self.__class__.__name__+' backward pass grad input['+str(i+1)+'] occured nan!!', grad_input[i].size())
    else:
        if isnan(grad_input.max()):
            print('Inside '+self.__class__.__name__+' backward pass grad input occured nan!!', grad_input.size())
    if type(grad_output) is tuple:
        for i in range(len(grad_output)):
            if type(grad_output[i]) == type(None):
                continue
            if isnan(grad_output[i].max()):
                print('Inside '+self.__class__.__name__+' backward pass grad output['+str(i+1)+'] occured nan!!', grad_output[i].size())
    else:
        if isnan(grad_output.max()):
            print('Inside '+self.__class__.__name__+' backward pass grad output occured nan!!'.grad_output.size())
