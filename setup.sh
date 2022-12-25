# download the Penn-Fudan dataset
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
# extract it in the current folder
unzip PennFudanPed.zip
rm PennFudanPed.zip

pip install albumentations==0.4.6 pycocotools --quiet

# Clone TorchVision repo and copy helper files
git clone https://github.com/pytorch/vision.git
cp vision/references/detection/utils.py ./
cp vision/references/detection/transforms.py ./
cp vision/references/detection/coco_eval.py ./
cp vision/references/detection/engine.py ./
cp vision/references/detection/coco_utils.py ./

git config --global user.name "ram-ch"
git config --global email