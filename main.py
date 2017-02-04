image_dir = './images'
out_image_dir = './out_images'
out_model_dir = './out_models'

nz = 100
batch_size = 100
n_epoch = 1e4
n_train = 2e5
image_save_interval = 5e4

fs = os.listdir(image_dir)

dataset = []
for fn in fs:
  with open('%s/%s' % (image_dir, fn), 'rb') as f:
    img_bin = f.read()
    dataset.append(img_bin)

