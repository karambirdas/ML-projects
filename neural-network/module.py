import math
class Value:
    #_child is an important variable, underscore because it does not change under any operation
    #__variable meaning it is protected like password and it can not be accessed
    def __init__(self, data, _children = (), _op = '', label =''):
           self.data = data
           self.grad = 0
           self._op = _op
           self._backward = lambda: None
           self._prev = set(_children)
           self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data + other.data), (self, other), '+')
        #we are adding _backward because it is exclusive to this 'add' operation, it does not mix with other backward function of other operations
        def _backward():
             self.grad += out.grad * 1.0
             other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    
    def __radd__(self, other):
         return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data * other.data), (self, other), '*')
        def _backward():
             self.grad += out.grad * other.data
             other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other 

    def __neg__(self):
         return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
         return self - other

    def __pow__(self, other):
         assert isinstance(other, (int, float)), "only supported for int/float"
         out = Value(self.data**other, (self,), f'**{other}')
         def _backward():
              self.grad += other*(self.data**(other - 1)) * out.grad
         out._backward = _backward
         return out

    def __truediv__(self, other):
         return self * other**-1

    #sigmoid function is too sensative for the activation function
    def sigmoid(self):
         x = self.data
         val = 1.0 / (1 + math.exp(-x))
         out = Value(val, (self,), 'Sigma')
         #No need to multiply with out.grad in the backward function. If it is multiplied then we have initialise the other.grad to be 1.
         #In our current formula other.grad won't matter.
         def _backward():
              self.grad += ((out.data)**2) * math.exp(- self.data) * out.grad
         out._backward = _backward
         return out
    
    def tanh(self):
        x = math.exp(2 * (self.data))
        t = (x-1) / (x+1)
        out = Value(t , (self,), 'Tanh')
        def _backward():
             self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_top(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_top(child)
                topo.append(v)
        build_top(self)

        self.grad = 1.0
        for i in reversed(topo):
             i._backward()
    
