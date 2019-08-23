from pip._internal import main as pipmain

if __name__ == '__main__':
    from sys import platform

    linux_packages = ['tensorflow-gpu==2.0.0-beta1']
    macos_packages = ['tensorflow==2.0.0-beta1']
    main_packages = [
        'mlagents-envs==0.9.1',
        'tensorboard==1.14.0',
        'numpy==1.14.5',
        'protobuf',
        'gym',
        'tqdm',
        'google-cloud-bigtable',
        'google-cloud-storage',
        'wandb'
    ]

    if platform.startswith('linux'):
        all_packages = linux_packages + main_packages
    elif platform == 'darwin':
        all_packages = macos_packages + main_packages

    for package in all_packages:
        pipmain(['install', package])