__author__ = 'kh'

from nn.rae import RecursiveAutoencoder
from numpy import arange, dot, exp, zeros, zeros_like, tanh, concatenate
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
        self.rae = rae

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
    def forward(self):
        embsize = self.rae.get_embsize()
        prePhrase = self.rae.encode(embsize)
        aftPhrase = self.rae.encode(embsize)

        output1 = dot(concatenate(prePhrase, aftPhrase), self.W1) + self.b1
        output2 = dot(concatenate(prePhrase, aftPhrase), self.W2) + self.b2

        output1 = exp(output1)/(exp(output1) + exp(output2))
        output2 = 1-output1

        softmaxLayer = []
        softmaxLayer.append(output1)
        softmaxLayer.append(output2)

        return softmaxLayer