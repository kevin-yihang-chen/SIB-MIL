import pandas as pd
import glob
annotation_list = glob.glob('annotation_tif/*.tif')
slide = []
label = []
for annotation in annotation_list:
    slide_name = annotation.split('/')[-1][:8]
    if 'test' in slide_name:
        slide.append(f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v0/testing/{slide_name}')
    else:
        slide.append(f'/data1/WSI/Patches/Features/Camelyon16/simclr_files_256_v0/training/{slide_name}')
    label.append(1)
df = pd.DataFrame({ 'label': label,'slide': slide})
df.to_csv('annotation_tif/samples.csv', index=False)