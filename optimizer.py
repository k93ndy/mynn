class SGD_basic:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        # for name, layer in layers.items():
        #     if 'affline_tier' in name:
        #         layer.W = layer.W - layer.dW*self.learning_rate
        #         layer.b = layer.b - layer.db*self.learning_rate
        layer.W = layer.W - layer.dW*self.learning_rate
        layer.b = layer.b - layer.db*self.learning_rate