import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat  
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import tqdm as tqdm




class PhysioNetDataset(Dataset):

    def __init__(self, input_folder, max_rows=259520, num_samples_to_load=320):

        self.input_folder = input_folder
        self.file_list = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.endswith('.csv')]
        self.max_rows = max_rows
        self.batches = []
        self.labels = []
        self.num_samples_to_load = num_samples_to_load

        self.randomize_data()

    def randomize_data(self):

        len_packets = (self.num_samples_to_load//3) + 1
        len_packet = 160 
        num_files_total = 109 

        for e in tqdm.tqdm( range(len_packets) ):

            file_name = self.file_list[np.random.randint(0, num_files_total )]
            file_path = os.path.join(self.input_folder, file_name)

            df = pd.read_csv(file_path)#.head(self.max_rows)

            
            for label_i in range(3):  
                df_label = df[df['label'] == label_i]
         
                if len(df_label) >= len_packet:

                    random_samples = df_label.sample(n=len_packet, random_state=42)

                    samples = random_samples.drop(columns=['label']).to_numpy()

                    self.batches.append(samples)  
                    self.labels.extend([label_i] * len_packet) 

                else:
                    print(f"Pas assez d'échantillons pour le label {label_i} (trouvés : {len(df_label)})")



    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):

        batch = torch.tensor(self.batches[idx], dtype=torch.float32)  
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return batch, label