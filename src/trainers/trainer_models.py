
# Trainer appelle les routines, les dataloaders et les models


from src.models.eggs_skeleton import EGGS_1 # EEGNET loader
from src.data.dataloader import PhysioNetDataset#BCIDataset, GDFDataset # BCI loader


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm as tqdm

import numpy as np

class EGGS_1_Trainer():

    def __init__(self, config):

        self.n_classes = config.n_classes

        self.model = EGGS_1(self.n_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.criterion = nn.CrossEntropyLoss()

    def load_train_dataset(self, batch_size=16, nb_data_to_load=3000, data_path = './src/data/datasources/physionet-extracted/'):
        dataset = PhysioNetDataset(data_path, num_samples_to_load=nb_data_to_load)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= 5: 
                break
            print(f"Batch {batch_idx}: Inputs shape {inputs.shape}, Labels shape {labels.shape}")
        print(f"Data training loaded : {len(dataloader)}")
        return dataloader



    def fit(self, train_set, num_epochs):

        epoch_loss = []

        for epoch in tqdm.tqdm( range(num_epochs) ):

            

            self.model.train()
            self.optimizer.zero_grad()
            
            for batch_idx, (inputs, labels) in enumerate(train_set):

                loss = []

                self.optimizer.zero_grad()

                for batch_n in range(inputs.shape[0]): 

                    output = self.model(inputs[batch_n].unsqueeze(0))  # .unsqueeze(0) pour ajouter une dimension de batch

                    # Calcul de la perte pour cet échantillon
                    self.loss_object = self.criterion(output, labels[batch_n].unsqueeze(0))  # .unsqueeze(0) pour ajuster la forme

                    # Ajout de la perte de cet échantillon à la liste
                    loss.append(self.loss_object.item())

                    # Rétropropagation des gradients
                    self.loss_object.backward()

                    # Mise à jour des poids pour cet échantillon
                    self.optimizer.step()

                loss.append(self.loss_object.item())


                # Validation stage a dev

            # Send data to tensorboard
            print(f"Epoch {epoch+1}, Loss: {np.mean(loss)}")

            epoch_loss.append( np.mean(loss) )

        return epoch_loss

                






