pip install gdown

DATA=coop_data/
mkdir $DATA
# DATA=/work/tianjun/few-shot-learning/prompt-moe/CoOp/data/
cd $DATA

# pip install gdown

mkdir -p caltech-101
cd caltech-101
# wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz 
wget https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip
unzip caltech-101.zip
mv caltech-101/101_ObjectCategories.tar.gz .
gdown 1hyarUivQE36mY6jSomru6Fjd-JzwcCzN
tar -xvf 101_ObjectCategories.tar.gz 
cd $DATA

mkdir -p oxford_pets
cd oxford_pets
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
gdown  1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs
tar -xvf images.tar.gz
tar -xvf annotations.tar.gz
cd $DATA

mkdir -p stanford_cars
cd stanford_cars
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz 
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat
gdown  1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT
tar -xvf cars_train.tgz 
tar -xvf cars_test.tgz 
tar -xvf car_devkit.tgz 
cd $DATA

mkdir -p oxford_flowers
cd oxford_flowers
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz 
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
gdown 1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0
gdown 1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT
tar -xvf 102flowers.tgz 
cd $DATA


wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xvf food-101.tar.gz
cd food-101
gdown 1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl
cd $DATA

wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar -xvf fgvc-aircraft-2013b.tar.gz
mv fgvc-aircraft-2013b/data fgvc_aircraft
cd $DATA

mkdir -p sun397
cd sun397
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
gdown 1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq
tar -xvf SUN397.tar.gz
unzip Partitions.zip
cd $DATA


wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvf dtd-r1.0.1.tar.gz
cd dtd
gdown 1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x
cd $DATA

mkdir -p eurosat
cd eurosat
wget http://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip EuroSAT.zip
gdown 1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o
cd $DATA

mkdir -p ucf101
cd ucf101
gdown  10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O
unzip UCF-101-midframes.zip
gdown 1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y
cd $DATA


# mkdir -p imagenet/images
# cd imagenet/images
# ##1. Download the data
# #get ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# #get ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

# ## 2. Extract the training data:
# mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
# tar -xvf ILSVRC2012_img_train.tar && mv ILSVRC2012_img_train.tar ../
# find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# cd ..

# ## 3. Extract the validation data and move images to subfolders:
# mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar && mv ILSVRC2012_img_val.tar ../
# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
# cd $DATA

# ## 4. Move the classname.txt to /imagenet/images
# cd ../scripts/
# mv classnames.txt ../coop_data/imagenet/images