
# Runner appelle le trainer


import torch


from src.trainers.trainer_models import EGGS_1_Trainer

#import config

from src.config.config_files import config_EEGS_1

# exec trainer with config


if __name__ == "__main__":

    config = config_EEGS_1()
    print("Config OK")
    trainer = EGGS_1_Trainer(config)

    train_set = trainer.load_train_dataset(batch_size = 16, nb_data_to_load=3600)

    num_epochs = 10
    print("run trainer fit")
    total_loss = trainer.fit(train_set, num_epochs)

    print("end")

