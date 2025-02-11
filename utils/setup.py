import os
import platform
import subprocess
import time

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

MAJOR = 0
MINOR = 5
PATCH = 0
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)

def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


def make_cuda_ext(name, module, sources):

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
                "-allow-unsupported-compiler",
            ]
        })



def get_ext_modules():
    ext_modules = []
    # only windows visual studio 2013+ support compile c/cuda extensions
    # If you force to compile extension on Windows and ensure appropriate visual studio
    # is intalled, you can try to use these ext_modules.
    force_compile = True
    if platform.system() != 'Windows' or force_compile:
        ext_modules = [
            make_cuda_ext(
                name='cpu_nms',
                module='nms',
                sources=['cpu_nms.cpp'],
                ),
            make_cuda_ext(
                name='gpu_nms',
                module='nms',
                sources=['gpu_nms.cu', 'nms_kernel.cu']),
        ]
    return ext_modules


def get_install_requires():
    install_requires = [
        'six', 'terminaltables',
        'opencv-python', 'matplotlib', 'visdom',
        'tqdm', 'tensorboardx', 'easydict',
        'pyyaml', 'halpecocotools',
        'torch>=1.1.0', 'torchvision>=0.3.0',
        'munkres', 'natsort'
    ]
    # official pycocotools doesn't support Windows, we will install it by third-party git repository later
    if platform.system() != 'Windows':
        install_requires.append('pycocotools')
    return install_requires


def is_installed(package_name):
    from pip._internal.utils.misc import get_installed_distributions
    for p in get_installed_distributions():
        if package_name in p.egg_name():
            return True
    return False


if __name__ == '__main__':
    setup(
        name='nms',
        description='Code for nms',
        keywords='computer vision, human pose estimation',
        url='https://github.com/MVIG-SJTU/AlphaPose',
        packages=find_packages(exclude=('data', 'exp',)),
        package_data={'': ['*.json', '*.txt']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        license='GPLv3',
        python_requires=">=3",
        setup_requires=['pytest-runner', 'numpy', 'cython'],
        tests_require=['pytest'],
        install_requires=get_install_requires(),
        ext_modules=get_ext_modules(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
