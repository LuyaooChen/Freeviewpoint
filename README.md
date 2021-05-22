# Freeviewpoint

![demo1](https://github.com/LuyaooChen/Freeviewpoint/blob/master/demo1.gif)

![demo2](https://github.com/LuyaooChen/Freeviewpoint/blob/master/demo2.gif)

## 介绍
自由视点视频软件，DIBR方法的CUDA实现。利用双参考视点合成虚拟视点，并进行图像空洞的修复、伪影去除等工作。有较好的处理速度和图像质量。还提供了对应的CPU实现版本，主要用作速度对比。

## 依赖
OpenCV3及以上 ，编译时需要添加CMAKE的 WITH_CUDA选项。在3.4,4.0版本进行过测试。
CUDA9及以上。在9.0,10.0进行过测试。

PS. CPU版本只需OpenCV

## 数据集
Ballet, Breakdancers, Newspaper, PoznanStreet，PoznanHall2，BookArrival等。
需要相机标定参数和最大最小深度值（Zmax，Zmin或叫 Znear，Zfar）
