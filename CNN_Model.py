from keras import layers, models


class CNN(object):
    def __init__(self, num_filter, kernel_size):
        # 创建一个Sequential模型
        self.model = models.Sequential()

        # 向模型中添加一个卷积层，指定输入图像的形状为 (28, 28, 1)，即28x28像素的灰度图像。
        self.model.add(
            layers.Conv2D(num_filter, (kernel_size, kernel_size), activation='relu', input_shape=(28, 28, 1)))

        # 添加一个最大池化层，用于减小特征图的空间尺寸，这里的池化窗口大小为 (2, 2)。
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(num_filter * 2, (kernel_size, kernel_size), activation='relu'))
        self.model.add(layers.MaxPooling2D((kernel_size, kernel_size)))
        self.model.add(layers.Conv2D(num_filter * 2, (kernel_size, kernel_size), activation='relu'))

        # 添加一个展平层，将卷积层输出的特征图展平成一个一维向量，以便连接到全连接层。
        self.model.add(layers.Flatten())

        # 添加一个全连接层，具有64个神经元
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

        # 编译模型，优化器为'RMSprop'，损失函数为'categorical_crossentropy'（用于多类分类任务），并指定评估指标为准确率（'acc'）
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()

    def fit(self, x, y, batch_size=64, epochs=60, path=None):
        # validation_split: 用于验证集划分的比例，这里设置为0.2表示将20%的训练数据用作验证集。
        # 这个参数用于在每个训练周期结束时评估模型性能，以便监测过拟合等问题。
        hist = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
        self.model.save(path + f"CNN.h5")
        return hist

    def evaluate(self, x, y):
        test_loss, test_acc = self.model.evaluate(x, y)
        return test_loss, test_acc