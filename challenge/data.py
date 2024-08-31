# Data Loaders for NeuroBench Challenge

import numpy as np
import torch
import stork
from neurobench.datasets.primate_reaching import PrimateReaching
from neurobench.datasets.utils import download_url
from urllib.error import URLError

import os

import logging

logger = logging.getLogger(__name__)
SAMPLING_RATE = 4e-3


def get_dataloader(cfg, dtype=torch.float32):

    dataloader = DatasetLoader(
        basepath=cfg.data.data_dir,
        ratio_val=cfg.data.ratio_val,
        random_val=cfg.data.random_val,
        extend_data=cfg.data.extend_data,
        sample_duration=cfg.data.sample_duration,
        remove_segments_inactive=cfg.data.remove_segments_inactive,
        p_drop=cfg.data.p_drop,
        p_insert=cfg.data.p_insert,
        jitter_sigma=cfg.data.jitter_sigma,
        dtype=dtype
    )

    return dataloader


def compute_input_firing_rates(data, cfg):

    mean1 = 0
    mean2 = 0

    for i in range(len(data)):
        mean1 += torch.sum(data[i][0][:, :96]) / cfg.data.sample_duration / 96
        try:
            mean2 += torch.sum(data[i][0][:, 96:]) / cfg.data.sample_duration / 96
        except:
            continue

    mean1 /= len(data)
    mean2 /= len(data)

    # For LOCO
    if data[0][0].shape[1] == 192:
        return mean1, mean2

    # FOR INDY
    else:
        return mean1, None


class PretrainPrimateReaching(PrimateReaching):
    """
    Load more sessions as dataset for the Primate Reaching Task with modified MD5 checksums.
    """

    def __init__(
        self,
        file_path,
        filename,
        num_steps,
        train_ratio=0.8,
        label_series=False,
        biological_delay=0,
        spike_sorting=False,
        stride=0.004,
        bin_width=0.028,
        max_segment_length=2000,
        split_num=1,
        remove_segments_inactive=False,
        download=True,
    ):

        super().__init__(
            file_path,
            filename,
            num_steps,
            train_ratio,
            label_series,
            biological_delay,
            spike_sorting,
            stride,
            bin_width,
            max_segment_length,
            split_num,
            remove_segments_inactive,
            download,
        )

    def download(self):
        """Download the Primate Reaching data if it doesn't exist already."""

        if self.filename in self.md5s.keys():
            md5 = self.md5s[self.filename]
        else:
            md5 = None

        if self._check_exists(self.file_path, md5):
            return

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # download file
        url = f"{self.url}{self.filename}"
        try:
            print(f"Downloading {url}")
            download_url(url, self.file_path, md5=md5)
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print()


class DatasetLoader:
    """Loads the data from the PrimateReaching dataset and splits it into train, val and test sets. The train and valid sets are split into samples of a given length, while the test set is kept as a single sample. The data is returned as a tuple of stork RasDatasets. This datasets can then be used as usual with the stork StandardGenerator."""

    def __init__(
        self,
        basepath,
        num_steps=1,
        dt=0.004,
        ratio_val=0.25,
        biological_delay=0,
        spike_sorting=False,
        label_series=False,
        random_val=False,
        extend_data=True,
        sample_duration=2,
        bin_width=None,
        stride=None,
        remove_segments_inactive=False,
        dtype=torch.float32,
        p_drop=0.0,
        p_insert=0.0,
        jitter_sigma=0.0,
    ):
        """Initialize

        Args:
            basepath (str): the path to the data folder
            filename (str): the name of the specific data file to load
            num_steps (int, optional): Argument for the Neurobench dataloader. Should be 1. Defaults to 1.
            dt (float, optional): Time step, should be 0.004 for the monkey data. Defaults to 0.004.
            ratio_val (list, optional): Ratio for validation set. Defaults to 0.25
            biological_delay (int, optional): Delay of readout w.r.t input. Defaults to 0.
            spike_sorting (bool, optional): If True, using single unit activities, otherwise multi unit activities. Defaults to False.
            label_series (bool, optional): Some neurobench argument. Just leave as is. Defaults to False.
            random_val (bool, optional): If True, samples the validation samples randomly from the train data. Otherwise takes the last samples. Defaults to False.
            extend_data (bool, optional): If true, extends the data to overlapping samples. Defaults to True.
            sample_duration (int, optional): sample duration in seconds. Defaults to 2.
            bin_width (_type_, optional): Some neurobench argument. Just leave as is. Defaults to None.
            stride (_type_, optional): Some neurobench argument. Just leave as is. Defaults to None.
            remove_segments_inactive (bool, optional): Some neurobench argument. Just leave as is. Defaults to False.
            dtype (_type_, optional): The dtype of the datasets. Defaults to torch.float32.
        """
        self.basepath = basepath
        self.num_steps = num_steps
        self.dt = dt
        self.ratio_val = ratio_val
        self.biological_delay = biological_delay
        self.spike_sorting = spike_sorting
        self.label_series = label_series
        self.random_val = random_val
        self.extend_data = extend_data
        self.sample_duration = sample_duration
        self.remove_segments_inactive = remove_segments_inactive
        self.dtype = dtype
        self.p_drop = p_drop
        self.p_insert = p_insert
        self.jitter_sigma = jitter_sigma

        if bin_width is None:
            self.bin_width = self.dt
        else:
            self.bin_width = bin_width
        if stride is None:
            self.stride = self.dt
        else:
            self.stride = stride

        self.n_time_steps = int(sample_duration / dt)

    def get_single_session_data(self, filename):
        dataset = PretrainPrimateReaching(
            file_path=self.basepath,
            filename=filename,
            num_steps=self.num_steps,
            train_ratio=0.5,  # Hardcoded here for 25 % test split
            bin_width=self.bin_width,
            biological_delay=self.biological_delay,
            remove_segments_inactive=self.remove_segments_inactive,
        )

        # If we want to remove inactive segments, we need to load the data again
        # with remove_segments_inactive=False for the test set
        if self.remove_segments_inactive:
            dataset_test = PretrainPrimateReaching(
                file_path=self.basepath,
                filename=filename,
                num_steps=self.num_steps,
                train_ratio=0.5,  # Hardcoded here for 25 % test split
                bin_width=self.bin_width,
                biological_delay=self.biological_delay,
                remove_segments_inactive=False,
            )
        else:
            dataset_test = dataset

        """Loads data of a single session and returns a tuple of stork RasDatasets containing the train, val and test data.

        Returns:
            tuple of stork RasDatasets: Train, val and test datasets
        """

        # Sum train & validation data (75 %) and make own validation split
        ind_tv = dataset.ind_train + dataset.ind_val

        # Effective validation ratio = val_ratio / 0.75
        eff_ratio_val = self.ratio_val / 0.75

        n_val = int(np.round(len(dataset) * eff_ratio_val))

        if self.random_val:
            start_idx = np.random.choice(a=ind_tv[:-n_val], size=1)[0]
            ind_val = np.array(ind_tv[start_idx : start_idx + n_val])
            ind_train = np.array(sorted(set(ind_tv) - set(ind_val)))

        else:
            ind_train = np.array(ind_tv[:-n_val])
            ind_val = np.array(sorted(set(ind_tv) - set(ind_train)))

        spikes = dataset.samples.T
        labels = dataset.labels.T

        spikes_testdat = dataset_test.samples.T
        labels_testdat = dataset_test.labels.T

        self.ind_train = ind_train
        self.ind_val = ind_val
        self.ind_test = dataset_test.ind_test

        # split into train, val and test
        spikes_train = spikes[ind_train]
        spikes_val = spikes[ind_val]
        spikes_test = spikes_testdat[dataset_test.ind_test]

        labels_train = labels[ind_train]
        labels_val = labels[ind_val]
        labels_test = labels_testdat[dataset_test.ind_test]

        # split val and train data into single samples
        if self.extend_data:
            logger.info("Extending data...")
            train_data, train_labels = self.extend_spikes(
                spikes_train, labels_train, self.n_time_steps
            )
            val_data, val_labels = self.extend_spikes(
                spikes_val, labels_val, self.n_time_steps
            )
        else:
            train_data, train_labels = self.extend_spikes(
                spikes_train, labels_train, self.n_time_steps, chunks=99
            )
            val_data, val_labels = self.extend_spikes(
                spikes_val, labels_val, self.n_time_steps, chunks=99
            )

        test_data = [spikes_test]
        test_labels = [labels_test]

        test_data = torch.stack(test_data)
        test_labels = torch.stack(test_labels)
        
        # Get augmentation kwargs for training dataset
        if any([self.p_drop > 0, self.p_insert > 0, self.jitter_sigma > 0]):
            
            data_augmentation_kwargs = dict(
                data_augmentation=True,
                p_drop=self.p_drop, 
                p_insert=self.p_insert, 
                sigma_t=self.jitter_sigma
            )
        else:
            data_augmentation_kwargs = {}

        # make it ras datasets
        train_ras_data = self.to_ras(train_data, train_labels, 
                                     **data_augmentation_kwargs)
        val_ras_data = self.to_ras(val_data, val_labels)
        test_ras_data = self.to_ras(test_data, test_labels)

        return train_ras_data, val_ras_data, test_ras_data

    def get_multiple_sessions_data(self, filenames):
        """Loads data from multiple sessions and concatenates them into a single dataset (split in train, test and validation).

        Args:
            filenames (list): List of filenames to load (all files should be in the folder specified by basepath    )

        Returns:
            tuple of stork RasDatasets for train and validation and a list of test dataset (one dataset for each session)
        """

        ds_train, ds_valid, ds_test = [], [], []

        for filename in filenames:
            monkey_ds_train, monkey_ds_valid, monkey_ds_test = (
                self.get_single_session_data(filename)
            )
            ds_train.append(monkey_ds_train)
            ds_valid.append(monkey_ds_valid)
            ds_test.append(monkey_ds_test)

        dataset_train = torch.utils.data.ConcatDataset(ds_train)
        dataset_valid = torch.utils.data.ConcatDataset(ds_valid)

        return dataset_train, dataset_valid, ds_test

    def extend_spikes(self, spikes, labels, chunks="all", chunksize=100):
        """Given spike data and labels of the shape [time x neuron], it cuts it into overlapping samples of shape [samples x n_time_steps x neuron]"""

        if chunks == "all":
            chunks = self.n_time_steps

        extended_spikes = []
        extended_labels = []

        for t in range(0, chunks, chunksize):
            curr_spikes = spikes[t:]
            curr_labels = labels[t:]

            splitter = np.arange(
                self.n_time_steps, curr_spikes.shape[0], self.n_time_steps
            )

            extended_spikes += np.split(curr_spikes, splitter)[:-1]
            extended_labels += np.split(curr_labels, splitter)[:-1]

        extended_spikes = torch.stack(extended_spikes)
        extended_labels = torch.stack(extended_labels)

        return extended_spikes, extended_labels

    def to_ras(self, data, labels, **data_augmentation_kwargs):
        ras_data = [[[], []] for _ in data]

        for i, sample in enumerate(data):
            for j in range(sample.shape[-1]):
                spike_times = np.where(sample[:, j] == 1)[0].tolist()
                ras_data[i][0] += spike_times
                ras_data[i][1] += [j] * len(spike_times)
            ras_data[i] = torch.tensor(ras_data[i], dtype=self.dtype)
        
        monkey_ds_kwargs = dict(
            nb_steps=data.shape[-2], nb_units=data.shape[-1], time_scale=1.0
        )

        monkey_ds = stork.datasets.RasDataset(
            (ras_data, labels), dtype=self.dtype, 
            **monkey_ds_kwargs, **data_augmentation_kwargs
        )

        return monkey_ds
