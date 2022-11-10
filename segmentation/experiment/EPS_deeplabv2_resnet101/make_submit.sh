EXP='deeplabv2_test'

mkdir -p ./results/VOC2012/Segmentation/comp6_test_cls
cp  ../../data/VOCdevkit/results/VOC2012/Segmentation/${EXP}/* ./results/VOC2012/Segmentation/comp6_test_cls/
tar -zcvf results.tar.gz results
