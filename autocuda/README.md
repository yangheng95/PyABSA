# autocuda - Auto choose the cuda device having the largest free memory in Pytorch

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg) 
[![PyPI](https://img.shields.io/pypi/v/autocuda)](https://pypi.org/project/autocuda/)
[![PyPI_downloads](https://img.shields.io/pypi/dm/autocuda)](https://pypi.org/project/autocuda/)
![Repo Size](https://img.shields.io/github/repo-size/yangheng95/autocuda)


# Usage
## Install
```
pip install autocuda
```

## ready to use


```
from autocuda import auto_cuda_info, auto_cuda_index, auto_cuda, auto_cuda_name

cuda_info_dict = auto_cuda_info()

cuda_device_index = auto_cuda_index()  # return cuda index having largest free memory. return 'cpu' if not cuda
# os.environ['CUDA_VISIBLE_DEVICES'] = [str(cuda_device_index)]

cuda_device = auto_cuda()
# model.to(cuda_device) # assume you have inited your pytorch model

cuda_device_name = auto_cuda_name()
print('Choosing cuda device: {}'.format(cuda_device_name))

```

### Copyright
This package is based on https://github.com/QuantumLiu/tf_gpu_manager with MIT license
