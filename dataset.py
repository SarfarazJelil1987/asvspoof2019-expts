import numpy as np
import torch
from torch.utils.data import Dataset
import pickle5 as pickle
import os
from torch.utils.data.dataloader import default_collate

torch.set_default_tensor_type(torch.FloatTensor)

class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, path_to_protocol, part='train', feature='LFCC',
                 genuine_only=False, feat_len=750, padding='repeat'):
        self.access_type = access_type
        # self.ptd = path_to_database
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        # self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature = feature
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        #protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ 'eval_tsne' + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
            #self.tag = {"-": "-", "AA": "AA", "AB": "AB", "AC": "AC", "BA": "BA", "BB": "BB", "BC": "BC", "CA": "CA", "CB":"CB", "CC":"CC"}
        self.label = {"spoof": 1, "bonafide": 0}

        print(protocol)
        #print(self.tag)
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            if genuine_only:
                assert self.part in ["train", "dev"]
                if self.access_type == "LA":
                    num_bonafide = {"train": 2580, "dev": 2548}
                    self.all_info = audio_info[:num_bonafide[self.part]]
                else:
                    self.all_info = audio_info[:5400]
            else:
                self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        #speaker, filename, _, tag, label, folder = self.all_info[idx]
        speaker, filename, _, tag, label = self.all_info[idx]
        
        #print(self.ptf + '/'+ folder + '/' + filename + '.pkl')
        try:
            #with open(self.ptf + '/'+ folder + '/' + filename + '.pkl', 'rb') as feature_handle:
            with open(self.ptf  + '/' + filename + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)
                #feat_mat = feat_mat[257:,]
                
                #feat_mat = feat_mat[0:512,:]
                #mean_mat = np.mean(feat_mat, axis=0)
                #std_mat = np.std(feat_mat, axis=0)

                #out = np.subtract(feat_mat, mean_mat)
                #out2 = np.divide(out, std_mat)
                #feat_mat = out2
        
        except:
            # add this exception statement since we may change the data split
            def the_other(train_or_dev):
                assert train_or_dev in ["train", "dev"]
                res = "dev" if train_or_dev == "train" else "train"
                return res
            with open(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)
                #feat_mat = feat_mat[0:257,]
                #feat_mat =feat_mat[0:60,]
                

        feat_mat = torch.from_numpy(feat_mat)
        #feat_mat = np.transpose(feat_mat)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len-self.feat_len)
            feat_mat = feat_mat[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                feat_mat = padding(feat_mat, self.feat_len)
            elif self.padding == 'repeat':
                feat_mat = repeat_padding(feat_mat, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        
        #return feat_mat, filename, self.tag[tag], self.label[label],folder
        return feat_mat, filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)

def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


if __name__ == "__main__":
    # path_to_database = '/data/neil/DS_10283_3336/'  # if run on GPU
    #path_to_features = '/media/sarfaraz/CQT/LA/pkl_features/'  # if run on GPU
    #path_to_protocol = '/home/phd/sarfaraz/ASVSpoof_2019/data/LA/ASVspoof2019_LA_cm_protocols_new/'
    path = 1

