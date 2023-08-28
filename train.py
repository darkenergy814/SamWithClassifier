from trainer.train import Train

trainer = Train(train_dir='./dataset/medical_sinusitis/polygon/train/', dataset_type='polygon', device='0')
dataset = trainer.dataset()
train_encoder = trainer.encoder(epochs=2)
