__author__ = 'kh'

from nn.rae import RecursiveAutoencoder
from numpy import arange, dot, exp, zeros, zeros_like, tanh, concatenate, log
from vec.wordvector import WordVectors
from ioutil import unpickle, Reader, Writer
from nn.instance import Instance

class ReorderClassifer(object):

    def __init__(self, W1, W2, b1, b2, rae, f=tanh):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
        self.f = f

    @classmethod
    def build(self, theta, embsize):
        raeSize = RecursiveAutoencoder.compute_parameter_num(embsize)
        offset = raeSize

        W1 = theta[offset:offset+embsize].reshape(1, embsize)
        offset += embsize

        W2 = theta[offset:offset+embsize].reshape(1, embsize)
        offset += embsize

        b1 = theta[offset:offset+1].reshape(1,1)
        offset += 1

        b2 = theta[offset:offset+1].reshape(1,1)
        offset += 1

        return ReorderClassifer(W1, W2, b1, b2)

    @classmethod
    def forward(self, instance, prePhrase, aftPhrase):
        embsize = self.rae.get_embsize()
        prePhrase = prePhrase
        aftPhrase = aftPhrase

        output1 = dot(concatenate(prePhrase, aftPhrase), self.W1) + self.b1
        output2 = dot(concatenate(prePhrase, aftPhrase), self.W2) + self.b2

        output1 = exp(output1)/(exp(output1) + exp(output2))
        output2 = 1-output1

        softmaxLayer = []
        softmaxLayer.append(output1)
        softmaxLayer.append(output2)

        reo_error = 0
        if instance.order == 1:
            reo_error = -1.0 * log(softmaxLayer[0])
        else:
            reo_error = -1.0 * log(softmaxLayer[1])

        return softmaxLayer, reo_error

    @classmethod
    def backward(cls, softmaxLayer, order, prePhrase, aftPhrase, total_grad):
        delta_to_rae = softmaxLayer

        if order == 1:
            total_grad.gradW1 += dot(softmaxLayer[1], concatenate(prePhrase.T, aftPhrase.T))
            total_grad.gradb1 += softmaxLayer[1]
            delta_to_rae[0] = softmaxLayer[1]
            delta_to_rae[1] = 0
        else:
            total_grad.gradW2 += dot(softmaxLayer[0], concatenate(prePhrase.T, aftPhrase.T))
            total_grad.gradb2 += softmaxLayer[0]
            delta_to_rae[0] = 0
            delta_to_rae[1] = softmaxLayer[0]

        delta_to_rae = dot(delta_to_rae[0], cls.W1.T) + dot(delta_to_rae[1], cls.W2.T)
        embSize = len(delta_to_rae)/2

        return delta_to_rae[0:embSize], delta_to_rae[embSize:embSize*2]

    class Gradient(object):
        def __init__(self, rm):
            self.gradW1 = zeros_like(rm.W1)
            self.gradW2 = zeros_like(rm.W2)
            self.gradb1 = zeros_like(rm.b1)
            self.gradb2 = zeros_like(rm.b2)

        def to_row_vector(self):
            '''Place all the gradients in a row vector
            '''
            vectors = []
            vectors.append(self.gradW1.reshape(self.gradW1.size, 1))
            vectors.append(self.gradW2.reshape(self.gradW2.size, 1))
            vectors.append(self.gradb1)
            vectors.append(self.gradb2)
            return concatenate(vectors)[:, 0]

    def get_zero_gradients(self):
            return self.Gradient(self)