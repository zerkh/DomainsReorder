#-*- coding: utf-8 -*-
__author__ = 'kh'

from nn.rae import RecursiveAutoencoder
from numpy import arange, dot, exp, zeros, zeros_like, tanh, concatenate, log
from vec.wordvector import WordVectors
from ioutil import unpickle, Reader, Writer
from nn.instance import Instance
from numpy import linalg as LA
from functions import tanh_norm1_prime, sum_along_column
from vec.wordvector import WordVectors
from ioutil import unpickle, Reader, Writer



class ReorderClassifer(object):

    def __init__(self, W1, W2, b1, b2, f=tanh):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
        self.f = f

    @classmethod
    def build(self, theta, embsize, rae):
        offset = 0

        W1 = theta[offset:offset + embsize*2].reshape(1, embsize*2)
        offset += embsize*2

        W2 = theta[offset:offset + embsize*2].reshape(1, embsize*2)
        offset += embsize*2

        b1 = theta[offset:offset + 1].reshape(1, 1)
        offset += 1

        b2 = theta[offset:offset + 1].reshape(1, 1)
        offset += 1

        return ReorderClassifer(W1, W2, b1, b2)

    def forward(self, instance, prePhrase, aftPhrase, embsize):
        output1 = dot(self.W1, concatenate((prePhrase, aftPhrase))) + self.b1
        output2 = dot(self.W2, concatenate((prePhrase, aftPhrase))) + self.b2

        output1 = exp(output1) / (exp(output1) + exp(output2))
        output2 = 1 - output1

        softmaxLayer = []
        softmaxLayer.append(output1)
        softmaxLayer.append(output2)

        reo_error = 0
        if instance.order == 1:
            reo_error = -1.0 * log(softmaxLayer[0])
        else:
            reo_error = -1.0 * log(softmaxLayer[1])

        return softmaxLayer, reo_error

    def backward(cls, softmaxLayer, order, prePhrase, aftPhrase, total_grad):
        delta_to_rae = softmaxLayer

        if order == 1:
            total_grad.gradW1 -= concatenate((prePhrase, aftPhrase)).T * softmaxLayer[1]
            total_grad.gradb1 -= softmaxLayer[1]
            delta_to_rae[0] = -1 * softmaxLayer[1]
            delta_to_rae[1] = 0
        else:
            total_grad.gradW2 -= concatenate((prePhrase, aftPhrase)).T * softmaxLayer[0]
            total_grad.gradb2 -= softmaxLayer[0]
            delta_to_rae[0] = 0
            delta_to_rae[1] = -1 * softmaxLayer[0]

        delta_to_rae = delta_to_rae[0] * cls.W1.T + delta_to_rae[1] * cls.W2.T
        embSize = len(delta_to_rae) / 2

        return delta_to_rae[0:embSize], delta_to_rae[embSize:embSize * 2]

    @classmethod
    def compute_parameter_num(cls, embsize):
        '''Compute the parameter number of a reorder model

        Args:
          embsize: dimension of word embedding vector

        Returns:
          number of parameters
        '''
        sz = embsize * 2  # W1
        sz += embsize * 2  # W2
        sz += 1  # b1
        sz += 1  # b2
        return sz


    def get_weights_square(self):
        square = (self.W1 ** 2).sum()
        square += (self.W2 ** 2).sum()
        return square


    def get_bias_square(self):
        square = (self.b1 ** 2).sum()
        square += (self.b2 ** 2).sum()
        return square

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

        def __mul__(self, other):
            self.gradW1 *= other
            self.gradW2 *= other
            self.gradb1 *= other
            self.gradb2 *= other
            return self

    def get_zero_gradients(self):
        return self.Gradient(self)
