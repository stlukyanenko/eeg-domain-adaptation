import torch.utils.data as data
import mne
import numpy as np
import scipy.io

import braindecode.preprocessing


class SingleSubjectDataset(data.Dataset):
    def __init__(self, file_name="A01T", root_dir="dataset/2a/", domain_id=0, annotated_percentage = 1.0):
        mne_raw_data = mne.io.read_raw_gdf(root_dir + file_name + ".gdf", verbose = 0)
        mne_raw_data.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

        labelmat = scipy.io.loadmat(root_dir + file_name + ".mat")
        label_list = labelmat['classlabel'][:,0]
        
        self.domain_id = domain_id

        event_list =  mne.events_from_annotations(mne_raw_data, verbose = 0)[0]
        event_codes = mne.events_from_annotations(mne_raw_data, verbose = 0)[1]
        
        def select_events(event_list, event_id):
            mask = (event_list[:, 2] == event_id)
            return event_list[mask, :]

        # self.data = [(mne.filter.filter_data(mne_raw_data.get_data(start=event_start, stop=event_start+1125), 250, 0, 38, verbose = False), label_list[i]-1) for i, event_start in enumerate(np.sort(select_events(event_list,  event_codes['768'])[:, 0]))]
        
        all_data = mne_raw_data.get_data()
        all_data = mne.filter.filter_data(all_data, 250, 0, 38, verbose = False)
        all_data = braindecode.preprocessing.exponential_moving_standardize(all_data)
        self.data = [(all_data [:,event_start:event_start+1125], label_list[i]-1) for i, event_start in enumerate(np.sort(select_events(event_list,  event_codes['768'])[:, 0]))]

        if annotated_percentage < 1.0:
            self.data = [x if np.random.rand() < annotated_percentage else (x[0], -1)  for x in self.data ]

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return *self.data[idx], self.domain_id


class MultiSubjectDataset(data.Dataset):
    def __init__(self, file_names=["A01T", "A02T", "A03T", "A04T", "A05T", "A06T", "A07T", "A08T", "A09T"],
        domain_ids=[0,1,2,3,4,5,6,7,8],
        root_dir="dataset/2a/", transform=None):
        
        self.subdatasets = [SingleSubjectDataset(file_names[i], root_dir, domain_id=domain_ids[i])  for i in range(len(file_names))]
        self.total_length = sum([len(ds) for ds in self.subdatasets])
        
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        len_accum = 0
        
        for ds in self.subdatasets:
            if idx < len_accum + len(ds):
                return ds[idx - len_accum]
            else:
                len_accum += len(ds)
        
        return None
