### pytorch 转的ONNX模型

gan_engine.onnx

### 序列化引擎

gan_engine.trt

### 模型的输入

- batch_size: 1

- 输入有两个，第一个输入的维度是[1, 350]，其中1是batch_size，350表示整个场景的25个框的参数的flatten，公式：350 = 25 * (7+1+6)，其中25为一个场景中最多可能存在的groundtruth的个数，7是每个检测框的位置大小信息(xyzwlh和rot)，1是类别，6是点云在bbox六个面上的密度; 

  第二个输入的维度是[1,128]，其中1是batch_size，128是128个服从正态分布的数。

- 输出有三个，只需要关注大小为[1,140]的output，其中1是batch_size， 140是 = 20 * 7, 表示20个false positive box 的参数(不包括类别，具体含义顺序和输入一致)

注：模型的输入需要从simulator中获得，在测试代码中是固定了一个vector来测试的

### 运行

1 修改CmakeLists.txt中TensorRT的路径

2 make

3 cd build && ./main