from src.timeseries.utils.config import read_config
from src.timeseries.utils.dataset import download_datasets

if __name__ == '__main__':
    project = 'snp'
    dataset_cfg = read_config('download_datasets_s3', project)
    download_datasets(dataset_cfg, project)