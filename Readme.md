# GoogleNet 复现

创新点：

- Inception结构，能够融合不同尺度的特征
- 1x1的卷积核用于降维
- 丢弃全连接层，使用池化层，大大减少模型参数量
- 添加了两个辅助分类器帮助训练