本套程序对应的训练源码在https://github.com/yjh0410/YOWOv2
对应的文章是作者在2023年发布的。export_onnx文件夹里的是导出onnx文件的程序，
需要把里面的export_onnx.py放在训练源码的主目录，yowo.py放在训练源码的models/yowo文件夹里替换yowo.py
训练源码的仓库里提供了pth文件的下载链接，在下载完pth文件后就可以export_onnx.py导出onnx文件。

我这边已经导出的onnx文件在百度云盘，链接: https://pan.baidu.com/s/1zfdnb9c956kM6kMNFRGJkw 提取码: u513
onnx文件有UCF101-24数据集和AVA v2.2数据集训练出的这两种，在加载onnx文件的时候，注意指定相符的数据集类型。
