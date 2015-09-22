#-*- coding: utf-8 -*-
'''
Training example class
@author: lpeng
'''

class Instance(object):
  '''A reordering training example'''
  
  def __init__(self, words, freq=1):
    '''
    Args:
      words: numpy.array (an int array of word indices)
      freq: frequency of this training example
    '''
    self.words = words
    self.freq = freq

  def __str__(self):
    parts = []
    parts.append(' '.join([str(i) for i in self.words]))
    parts.append(str(self.freq))
    return ' ||| '.join(parts)
    
  @classmethod
  def parse_from_str(cls, line, word_vector):
    '''The format of the line should be like the following:
       src_word1, src_word2,..., src_wordn ||| freq
       freq is optional
    '''
    pos = line.find(' ||| ')
    words = [word_vector.get_word_index(word) for word in line[0:pos].split()]
    if pos >= 0:
      freq = int(line[pos+5:])
    else:
      freq = 1

    return Instance(words, freq)

class ReorderInstance(object):
    def __init__(self, preWords, aftWords, order):
        self.preWords = preWords
        self.aftWords = aftWords
        self.order = order

    @classmethod
    def paser_from_str(cls, line, word_vector):
        strs = line.split(' ct1=')
        order = 0
        if strs[0] == 'mono':
            order = 1

        strs = strs[1].split(' et1=')
        phrase = strs[0]
        preWords = [word_vector.get_word_index(word) for word in phrase]

        strs = strs[1].split(' ct2=')
        strs = strs[1].split(' et2=')
        phrase = strs[0]
        aftWards = [word_vector.get_word_index(word) for word in phrase]

        return ReorderInstance(preWords, aftWards, order)