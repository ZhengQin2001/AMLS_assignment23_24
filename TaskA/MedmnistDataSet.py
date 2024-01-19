import numpy as np
from medmnist.info import INFO, HOMEPAGE
from numpy.lib.npyio import NpzFile
from PIL import Image


class MedmnistDataSet:
    def __init__(self, datafile: NpzFile, flag: str, dataset_name: str):

        self.flag = flag
        self.dataset_name = dataset_name

        try:
            self.imgs, self.labels = datafile[f'{flag}_images'], datafile[f'{flag}_labels']
        except KeyError as e:
            print("Error:", e)

        self.__len__ = self.imgs.shape[0]
        self.info = INFO[self.dataset_name]

    def __getitems__(self, idx: int):      
        img, target = self.imgs[idx], self.labels[idx].astype(int)
        img = Image.fromarray(img)
        return img, target[0]
    
    def montage(self, length=20, replace=False, save_folder=None):
        from medmnist.utils import montage2d
        
        n_sel = length * length
        sel = np.random.choice(self.__len__, size=n_sel, replace=replace)

        montage_img = montage2d(imgs=self.imgs, n_channels=self.info['n_channels'], sel=sel)

        # Create a filename that includes the dataset name and flag
        filename = f"montage_{self.flag}_{self.dataset_name}.jpg"

        # Save the montage image in the specified folder (if provided) or in the current directory
        if save_folder:
            save_path = os.path.join(save_folder, filename)
        else:
            save_path = filename

        montage_img.save(save_path)

        return montage_img
