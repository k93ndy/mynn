class SGD_Basic:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update(self, layer, name=None):
        layer.W = layer.W - self.learning_rate*layer.dW
        layer.b = layer.b - self.learning_rate*layer.db

class SGD_Momentum:
    def __init__(self, learning_rate=0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = {}
    
    def update(self, layer, name):
        if not name in self.v:
            self.v[name] = {}
            self.v[name]['W'] = 0.
            self.v[name]['b'] = 0.
        self.v[name]['W'] = self.momentum*self.v[name]['W'] + self.learning_rate*layer.dW
        self.v[name]['b'] = self.momentum*self.v[name]['b'] + self.learning_rate*layer.db 
        layer.W = layer.W - self.v[name]['W']
        layer.b = layer.b - self.v[name]['b']