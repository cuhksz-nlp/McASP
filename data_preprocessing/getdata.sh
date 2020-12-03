############## process data ##############

# download Universal Dependencies 2.4
# If this step fails, you can manually download the file and put it under this directory
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/ud-treebanks-v2.4.tgz

tar zxvf ud-treebanks-v2.4.tgz
rm ud-treebanks-v2.4.tgz

## process UD
python data_preprocessing.py --dataset=ud --translate

## process CTB5
python data_preprocessing.py --dataset=ctb5

## process CTB6
python data_preprocessing.py --dataset=ctb6

## process CTB7
python data_preprocessing.py --dataset=ctb7

## process CTB9
python data_preprocessing.py --dataset=ctb9

############### process data ##############
