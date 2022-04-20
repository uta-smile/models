import pickle


# path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task003_Liver/nnUNetPlansv2.1_plans_3D.pkl"
path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task006_Lung/splits_final.pkl"
file=open(path,'rb')
plans = pickle.load(file)

print(plans)
