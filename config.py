import os, torch

DIR_PATH = os.path.dirname(__file__)

models_id_dict = {"mobilenet": 1, "resnet18": 2, "vgg16": 1, "resnet152": 4}

dataset_name = "caltech256"

# Standard Configuration of the Arguments 
split_ratio = 0.2
model_name = "mobilenet"
input_dim = 224
seed = 42 # the answer to life the universe and everything
cuda = True
distribution = "linear" 
exit_type = "bnpool"
batch_size_train = 256
batch_size_test = 1
pretrained = True
n_branches = 5
n_exits = n_branches + 1
max_iter = 200
#max_iter = 10
read_inf_data = True
a0 = 1
c = 1
alpha = 0.602
gamma = 0.101
threshold = 0.8
max_exits = 6
timeout = 3
temp_init = 1.5
step = 0.1
max_overhead = 30
step_overhead = 5
beta_start = 0
beta_end = 200
beta_step = 2
max_patience = 50
use_gpu = True

# Parameters of Data Augmentation
#To normalize the input images data, according to ImageNet dataset.
mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
std =  [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
h_flip_prob = 0.25
rotation_angle = 25
brightness = (0.80, 1.20)


root_dataset_path = os.path.join(DIR_PATH, "dataset")
root_indices_path = os.path.join(DIR_PATH, "indices")
resultsPath = os.path.join(DIR_PATH, "results")

dataset_path_dict = {"caltech256": os.path.join(root_dataset_path, "caltech256")}
idx_path_dict = {"caltech256": os.path.join(root_indices_path, "caltech256")}
nr_class_dict = {"caltech256": {3: 258, 5: 258, 1: 257}}
input_dim_dict = {3: [330, 300], 5: [256, 224], 1: [256, 224]}



DEBUG = True

#URLs
#Edge Device's URL
HOST_EDGE = "146.164.69.144"
#HOST_EDGE = "192.168.0.20"
PORT_EDGE = 5001
URL_EDGE = "http://%s:%s"%(HOST_EDGE, PORT_EDGE)
urlConfModelEdge = "%s/api/edge/modelConfiguration"%(URL_EDGE)



#Cloud server's URL
#HOST_CLOUD = "146.164.69.144"
HOST_CLOUD = "54.233.184.166"
#HOST_CLOUD = "192.168.0.20"
PORT_CLOUD = 3001
URL_CLOUD = "http://%s:%s"%(HOST_CLOUD, PORT_CLOUD)
urlConfModelCloud = "%s/api/cloud/modelConfiguration"%(URL_CLOUD)


threshold_list = [0.8]
overhead_list = [0, 1, 5, 10, 15, 20, 25, 30]
gamma_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


weight_decay=2e-05 
ngpus=6
max_epochs=600 
momentum=0.9
lr=0.5
lr_min = 0
lr_scheduler='cosineannealinglr' 
lr_warmup_epochs=5
lr_warmup_method='linear' 
lr_warmup_decay=0.01
norm_weight_decay=0.0
label_smoothing=0.1 
mixup_alpha=0.2 
cutmix_alpha=1.0 
auto_augment='ta_wide'
random_erase=0.1
  
ra_sampler=True
ra_reps=4
ra_magnitude = 9

# EMA configuration
model_ema=True 
model_ema_steps=32 
model_ema_decay=0.99998

# Resizing
interpolation='bilinear'
val_resize_size=232 
val_crop_size=224
train_crop_size=176
hflip_prob = 0.5


fixed_threshold = 0.6