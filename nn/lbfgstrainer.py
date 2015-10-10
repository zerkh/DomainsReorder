# -*- coding: utf-8 -*-
'''
Reordering classifier and related training code

@author: lpeng
'''
from __future__ import division
from sys import stderr
import argparse
import logging
import cPickle as pickle

from numpy import concatenate, zeros_like, zeros
from numpy.random import get_state, set_state, seed
from mpi4py import MPI
import random

from ioutil import Writer
from ioutil import getPhrases
from timeutil import Timer
import lbfgs
from ioutil import Reader
from nn.rae import RecursiveAutoencoder
from nn.reorder import ReorderClassifer
from nn.util import init_W
from nn.instance import Instance, ReorderInstance
from nn.signals import TerminatorSignal, WorkingSignal, ForceQuitSignal
from errors import GridentCheckingFailedError
from vec.wordvector import WordVectors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
worker_num = comm.Get_size()
rank = comm.Get_rank()


def send_terminate_signal():
    param = TerminatorSignal()
    comm.bcast(param, root=0)


def send_working_signal():
    param = WorkingSignal()
    comm.bcast(param, root=0)


def send_force_quit_signal():
    param = ForceQuitSignal()
    comm.bcast(param, root=0)


def preTrain(theta, instances, total_internal_node_num,
             word_vectors, embsize, lambda_rec, lambda_reg):
    '''Compute the value and gradients of the objective function at theta

    Args:
      theta: model parameter
      instances: training instances
      total_internal_node_num: total number of internal nodes
      embsize: word embedding vector size
      lambda_reg: the weight of regularizer

    Returns:
      total_cost: the value of the objective function at theta
      total_grad: the gradients of the objective function at theta
    '''

    if rank == 0:
        # send working signal
        send_working_signal()

        # send theta
        comm.Bcast([theta, MPI.DOUBLE], root=0)

        # init recursive autoencoder
        rae = RecursiveAutoencoder.build(theta, embsize)

        # compute local reconstruction error and gradients
        rec_error, gradient_vec = process_rae_local_batch(rae, word_vectors, instances)

        # compute total reconstruction error
        total_rec_error = comm.reduce(rec_error, op=MPI.SUM, root=0)

        # compute total cost
        reg = rae.get_weights_square()
        total_cost = lambda_rec * total_rec_error / total_internal_node_num + lambda_reg / 2 * reg

        # compute gradients
        total_grad = zeros_like(gradient_vec)
        comm.Reduce([gradient_vec, MPI.DOUBLE], [total_grad, MPI.DOUBLE],
                    op=MPI.SUM, root=0)
        total_grad /= total_internal_node_num

        for grad in total_grad:
            grad *= lambda_rec

        # gradients related to regularizer
        reg_grad = rae.get_zero_gradients()
        reg_grad.gradWi1 += rae.Wi1
        reg_grad.gradWi2 += rae.Wi2
        reg_grad.gradWo1 += rae.Wo1
        reg_grad.gradWo2 += rae.Wo2
        reg_grad *= lambda_reg

        total_grad += reg_grad.to_row_vector()

        return total_cost, total_grad
    else:
        while True:
            # receive signal
            signal = comm.bcast(root=0)
            if isinstance(signal, TerminatorSignal):
                return
            if isinstance(signal, ForceQuitSignal):
                exit(-1)

            # receive theta
            comm.Bcast([theta, MPI.DOUBLE], root=0)

            # init recursive autoencoder
            rae = RecursiveAutoencoder.build(theta, embsize)

            # compute local reconstruction error and gradients
            rec_error, gradient_vec = process_rae_local_batch(rae, word_vectors, instances)

            # send local reconstruction error to root
            comm.reduce(rec_error, op=MPI.SUM, root=0)

            # send local gradients to root
            comm.Reduce([gradient_vec, MPI.DOUBLE], None, op=MPI.SUM, root=0)


def compute_cost_and_grad(theta, instances, word_vectors, embsize, total_internal_node, lambda_rec, lambda_reg, lambda_reo, instances_of_News, is_Test):
    '''Compute the value and gradients of the objective function at theta

    Args:
    theta: model parameter
    instances: training instances
    embsize: word embedding vector size
    lambda_reg: the weight of regularizer
    lambda_reo: the weight of reo

    Returns:
    total_cost: the value of the objective function at theta
    total_grad: the gradients of the objective function at theta
    '''
    if rank == 0:
        # send work signal
        send_working_signal()

        # send theta
        comm.Bcast([theta, MPI.DOUBLE], root=0)

        if is_Test:
            instances_of_test, _ = prepare_test_data(word_vectors, instances_of_News)
            instances_of_test = random.sample(instances_of_test, 500)
            test(instances_of_test, theta, word_vectors, isPrint=True)

        #init rae
        rae = RecursiveAutoencoder.build(theta, embsize)

        offset = RecursiveAutoencoder.compute_parameter_num(embsize)

        rm = ReorderClassifer.build(theta[offset:], embsize, rae)

        #compute local reconstruction error, reo and gradients
        local_rae_error, local_rm_error,rae_rec_gradient, rae_gradient, rm_gradient, wordvector_gradient \
            = process_local_batch(rm, rae, word_vectors, instances, lambda_rec, lambda_reo)

        # compute total reconstruction error
        total_rae_error = comm.reduce(local_rae_error, op=MPI.SUM, root=0)
        total_rm_error = comm.reduce(local_rm_error, op=MPI.SUM, root=0)

        # compute total cost
        reg = rae.get_weights_square() + rm.get_weights_square()
        final_cost = total_rm_error / len(instances) + total_rae_error / total_internal_node + lambda_reg / 2 * reg

        # compute gradients
        #词向量未归一化,未加入正则化
        total_rae_rec_grad = zeros_like(rae_rec_gradient)
        total_rae_grad = zeros_like(rae_gradient)
        total_rm_grad = zeros_like(rm_gradient)
        total_wordvec_grad = zeros_like(wordvector_gradient)
        comm.Reduce([rae_rec_gradient, MPI.DOUBLE], [total_rae_rec_grad, MPI.DOUBLE],
                    op=MPI.SUM, root=0)
        comm.Reduce([rae_gradient, MPI.DOUBLE], [total_rae_grad, MPI.DOUBLE],
                    op=MPI.SUM, root=0)
        comm.Reduce([rm_gradient, MPI.DOUBLE], [total_rm_grad, MPI.DOUBLE],
                    op=MPI.SUM, root=0)
        comm.Reduce([wordvector_gradient, MPI.DOUBLE], [total_wordvec_grad, MPI.DOUBLE],
                    op=MPI.SUM, root=0)
        total_rae_grad /= len(instances)
        total_rm_grad /= len(instances)
        total_rae_rec_grad /= total_internal_node
        total_rae_grad += total_rae_rec_grad

        # gradients related to regularizer
        reg_grad = rae.get_zero_gradients()
        reg_grad.gradWi1 += rae.Wi1
        reg_grad.gradWi2 += rae.Wi2
        reg_grad.gradWo1 += rae.Wo1
        reg_grad.gradWo2 += rae.Wo2
        reg_grad *= lambda_reg

        total_rae_grad += reg_grad.to_row_vector()

        reg_grad = rm.get_zero_gradients()
        reg_grad.gradW1 += rm.W1
        reg_grad.gradW2 += rm.W2
        reg_grad.gradb1 += rm.b1
        reg_grad.gradb2 += rm.b2
        reg_grad *= lambda_reg

        total_rm_grad += reg_grad.to_row_vector()

        return final_cost, concatenate((total_wordvec_grad, total_rae_grad, total_rm_grad))
    else:
        while True:
            # receive signal
            signal = comm.bcast(root=0)
            if isinstance(signal, TerminatorSignal):
                return
            if isinstance(signal, ForceQuitSignal):
                exit(-1)

            # receive theta
            comm.Bcast([theta, MPI.DOUBLE], root=0)

            # init recursive autoencoder
            rae = RecursiveAutoencoder.build(theta, embsize)
            offset = RecursiveAutoencoder.compute_parameter_num(embsize)
            rm = ReorderClassifer.build(theta[offset:], embsize, rae)

            # compute local reconstruction error, reo and gradients
            local_rae_error, local_rm_error,rae_rec_gradient, rae_gradient, rm_gradient, wordvector_gradient \
                = process_local_batch(rm, rae, word_vectors, instances, lambda_rec, lambda_reo)

            # send local reconstruction error to root
            comm.reduce(local_rae_error, op=MPI.SUM, root=0)
            comm.reduce(local_rm_error, op=MPI.SUM, root=0)

            # send local gradients to root
            comm.Reduce([rae_rec_gradient, MPI.DOUBLE], None, op=MPI.SUM, root=0)
            comm.Reduce([rae_gradient, MPI.DOUBLE], None, op=MPI.SUM, root=0)
            comm.Reduce([rm_gradient, MPI.DOUBLE], None, op=MPI.SUM, root=0)
            comm.Reduce([wordvector_gradient, MPI.DOUBLE], None, op=MPI.SUM, root=0)


def process_rae_local_batch(rae, word_vectors, instances):
    gradients = rae.get_zero_gradients()
    total_rec_error = 0
    for instance in instances:
        words_embedded = word_vectors[instance.words]
        root_node, rec_error = rae.forward(words_embedded)
        rae.backward(root_node, gradients, freq=instance.freq)
        total_rec_error += rec_error * instance.freq

    return total_rec_error, gradients.to_row_vector()


def process_local_batch(rm, rae, word_vectors, instances, lambda_rec, lambda_reo):
    rae_rec_gradients = rae.get_zero_gradients()
    rae_gradients = rae.get_zero_gradients()
    rm_gradients = rm.get_zero_gradients()
    wordvectors_gradients = word_vectors.get_zero_gradients()
    total_rm_error = 0
    total_rae_error = 0
    for instance in instances:
        words_embedded = word_vectors[instance.preWords]
        root_prePhrase, rec_error = rae.forward(words_embedded)
        total_rae_error += lambda_rec * rec_error
        tmp_rae_gradients = rae.get_zero_gradients()

        words_embedded = word_vectors[instance.aftWords]
        root_aftPhrase, rec_error = rae.forward(words_embedded)
        total_rae_error += lambda_rec * rec_error

        rae.backward(root_prePhrase, tmp_rae_gradients)
        rae.backward(root_aftPhrase, tmp_rae_gradients)
        tmp_rae_gradients *= lambda_rec
        rae_rec_gradients += tmp_rae_gradients

        softmaxLayer, reo_error = rm.forward(instance, root_prePhrase.p, root_aftPhrase.p, embsize)
        total_rm_error += reo_error * lambda_reo
        delta_to_left, delta_to_right = rm.backward(softmaxLayer, instance.order, root_prePhrase.p, root_aftPhrase.p,
                                                    rm_gradients)
        tmp_rae_gradients = rae.get_zero_gradients()
        rae.backward(root_prePhrase, tmp_rae_gradients, wordvectors_gradients, delta_to_left, isRec=False)
        rae.backward(root_aftPhrase, tmp_rae_gradients, wordvectors_gradients, delta_to_right, isRec=False)
        tmp_rae_gradients *= lambda_reo
        rae_gradients += tmp_rae_gradients
    rm_gradients *= lambda_reo

    return total_rae_error, total_rm_error, rae_rec_gradients.to_row_vector(), \
           rae_gradients.to_row_vector(), rm_gradients.to_row_vector(), wordvectors_gradients.to_row_vector()


def init_theta(embsize, size_of_wordvector, _seed=None):
    if _seed != None:
        ori_state = get_state()
        seed(_seed)

    parameters = []

    #wordvectors
    parameters.append(init_W(embsize, size_of_wordvector))
    # Wi1
    parameters.append(init_W(embsize, embsize))
    # Wi2
    parameters.append(init_W(embsize, embsize))
    # bi
    parameters.append(zeros(embsize))

    # Wo1
    parameters.append(init_W(embsize, embsize))
    # Wo2
    parameters.append(init_W(embsize, embsize))
    # bo1
    parameters.append(zeros(embsize))
    # bo2
    parameters.append(zeros(embsize))

    parameters.append(init_W(1, embsize * 2))
    parameters.append(init_W(1, embsize * 2))
    parameters.append(zeros(1))
    parameters.append(zeros(1))

    if _seed != None:
        set_state(ori_state)

    return concatenate(parameters)


def prepare_rae_data(word_vectors=None, datafile=None):
    '''Prepare training data for rae
    Args:
      word_vectors: an instance of vec.wordvector
      datafile: location of data file

    Return:
      instances: a list of Instance
      word_vectors: word_vectors
      total_internal_node: total number of internal nodes
    '''
    if rank == 0:
        # broadcast word vectors
        comm.bcast(word_vectors, root=0)

        instance_strs = []
        # load raw data
        for file in datafile:
            with Reader(file) as file:
                for line in file:
                    phrases = getPhrases(line)
                    instance_strs.append(phrases[0])
                    instance_strs.append(phrases[1])

        # send training data
        instance_num = len(instance_strs)
        esize = int(instance_num / worker_num + 0.5)
        sizes = [esize] * worker_num
        sizes[-1] = instance_num - esize * (worker_num - 1)
        offset = sizes[0]
        for i in range(1, worker_num):
            comm.send(instance_strs[offset:offset + sizes[i]], dest=i)
            offset += sizes[i]
        comm.barrier()

        local_instance_strs = instance_strs[0:sizes[0]]
        del instance_strs

        instances, internal_node_num = load_rae_instances(local_instance_strs,
                                                          word_vectors)
        total_internal_node = comm.allreduce(internal_node_num, op=MPI.SUM)
        return instances, word_vectors, total_internal_node
    else:
        word_vectors = comm.bcast(root=0)

        # receive data
        local_instance_strs = comm.recv(source=0)
        comm.barrier()

        instances, internal_node_num = load_rae_instances(local_instance_strs,
                                                          word_vectors)
        total_internal_node = comm.allreduce(internal_node_num, op=MPI.SUM)
        return instances, word_vectors, total_internal_node


def prepare_test_data(word_vectors=None, dataFile=None):
    instance_lines = []
    with Reader(dataFile) as file:
        for line in file:
            instance_lines.append(line)
    instances = load_instances(instance_lines, word_vectors)

    return instances, word_vectors

def prepare_data(word_vectors=None, dataFile=None):
    '''Prepare training data
    Args:
    word_vectors: an instance of vec.wordvector
    dataFile: raw training file

    Return:
    instances: a list of ReorderInstance
    word_vectors
    '''
    if rank == 0:
        # broadcast word_vectors
        comm.bcast(word_vectors, root=0)

        instance_lines = []

        for file in dataFile:
            with Reader(file) as file:
                for line in file:
                    instance_lines.append(line)

        instance_num = len(instance_lines)
        esize = int(instance_num / worker_num + 0.5)
        sizes = [esize] * worker_num
        sizes[-1] = instance_num - esize * (worker_num - 1)
        offset = sizes[0]
        # send training data
        for i in range(1, worker_num):
            comm.send(instance_lines[offset:offset + sizes[i]], dest=i)
            offset += sizes[i]
        comm.barrier()

        local_instance_lines = instance_lines[0:sizes[0]]
        del instance_lines

        instances = load_instances(local_instance_lines, word_vectors)

        return instances, word_vectors
    else:
        word_vectors = comm.bcast(root=0)

        # receive data
        instance_lines = comm.recv(source=0)
        comm.barrier()

        instances = load_instances(instance_lines, word_vectors)

        return instances, word_vectors


def load_rae_instances(instance_strs, word_vectors):
    '''Load rae training examples

    Args:
      instance_strs: each string is a training example
      word_vectors: an instance of vec.wordvector

    Return:
      instances: a list of Instance
    '''

    instances = [Instance.parse_from_str(i, word_vectors) for i in instance_strs]

    instances = [i for i in instances if len(i.words) != 0]
    total_internal_node = 0
    for instance in instances:
        total_internal_node += (len(instance.words) - 1) * instance.freq
    return instances, total_internal_node


def load_instances(instances_lines, word_vectors):
    '''Load real training examples

    Args:
        instance_lines: each string is a training example
        word_vectors: an instance of vec.wordvector

    Return:
        instances: a list of ReorderInstance
    '''
    instances = [ReorderInstance.paser_from_str(i, word_vectors) for i in instances_lines]

    instances = [i for i in instances if len(i.preWords) != 0 and len(i.aftWords) != 0]

    return instances


def test(instances, theta, word_vectors, isPrint=False):
    if isPrint:
        outfile = open('./output/test_result.txt', 'w')
    total_lines = len(instances)
    total_true = 0

    # init rae
    rae = RecursiveAutoencoder.build(theta, embsize)

    offset = RecursiveAutoencoder.compute_parameter_num(embsize)

    rm = ReorderClassifer.build(theta[offset:], embsize, rae)

    for instance in instances:
        words_embedded = word_vectors[instance.preWords]
        root_prePhrase, rec_error = rae.forward(words_embedded)

        words_embedded = word_vectors[instance.aftWords]
        root_aftPhrase, rec_error = rae.forward(words_embedded)

        softmaxLayer, reo_error = rm.forward(instance, root_prePhrase.p, root_aftPhrase.p, embsize)

        if instance.order == 1 and softmaxLayer[0] >= softmaxLayer[1]:
            total_true += 1
        if instance.order == 0 and softmaxLayer[0] < softmaxLayer[1]:
            total_true += 1

        if isPrint:
            outfile.write("%f\t[%f,%f]\n" % (instance.order, softmaxLayer[0], softmaxLayer[1]))

    if isPrint:
        outfile.write("Total instances: %f\tTotal true predictions: %f\t" % (total_lines, total_true))
        outfile.write("Precision: %f" % (float(total_true / total_lines)))
    print("Total instances: %f\tTotal true predictions: %f\t" % (total_lines, total_true))
    print("Precision: %f" % (float(total_true / total_lines)))


class ThetaSaver(object):
    def __init__(self, model_name, every=1):
        self.idx = 1
        self.model_name = model_name
        self.every = every

    def __call__(self, xk):
        if self.every == 0:
            return

        if self.idx % self.every == 0:
            model = self.model_name
            pos = model.rfind('.')
            if pos < 0:
                filename = '%s.iter%d' % (model, self.idx)
            else:
                filename = '%s.iter%d%s' % (model[0:pos], self.idx, model[pos:])

            with Writer(filename) as writer:
                [writer.write('%20.8f\n' % v) for v in xk]
        self.idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-instances_of_Education', required=False)
    parser.add_argument('-instances_of_Laws', required=False)
    parser.add_argument('-instances_of_Thesis', required=False)
    parser.add_argument('-instances_of_Science', required=False)
    parser.add_argument('-instances_of_Spoken', required=False)
    parser.add_argument('-isTest', type=int, default=1)
    parser.add_argument('-instances_of_News', required=False)
    parser.add_argument('-model', required=True,
                        help='model name')
    parser.add_argument('-word_vector', required=True,
                        help='word vector file', )
    parser.add_argument('-lambda_reg', type=float, default=0.15,
                        help='weight of the regularizer')
    parser.add_argument('-lambda_reo', type=float, default=0.15,
                        help='weight of the reo')
    parser.add_argument('-lambda_rec', type=float, default=0.15,
                        help='weight of the rec')
    parser.add_argument('--save-theta0', action='store_true',
                        help='save theta0 or not, for dubegging purpose')
    parser.add_argument('--checking-grad', action='store_true',
                        help='checking gradients or not, for dubegging purpose')
    parser.add_argument('-m', '--maxiter', type=int, default=100,
                        help='max iteration number', )
    parser.add_argument('-e', '--every', type=int, default=0,
                        help='dump parameters every --every iterations', )
    parser.add_argument('--seed', default=None,
                        help='random number seed for initialize random', )
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        help='verbose level')
    options = parser.parse_args()

    instances_files = []
    if options.instances_of_Science != None:
        instances_files.append(options.instances_of_Science)
    if options.instances_of_Spoken != None:
        instances_files.append(options.instances_of_Spoken)
    if options.instances_of_Thesis != None:
        instances_files.append(options.instances_of_Thesis)
    if options.instances_of_Education != None:
        instances_files.append(options.instances_of_Education)
    if options.instances_of_Laws != None:
        instances_files.append(options.instances_of_Laws)

    model = options.model
    word_vector_file = options.word_vector
    lambda_reg = options.lambda_reg
    lambda_reo = options.lambda_reo
    lambda_rec = options.lambda_rec
    save_theta0 = options.save_theta0
    checking_grad = options.checking_grad
    maxiter = options.maxiter
    every = options.every
    _seed = options.seed
    verbose = options.verbose
    is_Test = options.isTest
    instances_of_News = options.instances_of_News

    if rank == 0:
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        if checking_grad:
            logger.setLevel(logging.WARN)
        else:
            logger.setLevel(logging.INFO)

        print >> stderr, 'Instances file: %s' % instances_files
        print >> stderr, 'Model file: %s' % model
        print >> stderr, 'Word vector file: %s' % word_vector_file
        print >> stderr, 'lambda_reg: %20.18f' % lambda_reg
        print >> stderr, 'lambda_rec: %20.18f' % lambda_rec
        print >> stderr, 'lambda_reo: %20.18f' % lambda_reo
        print >> stderr, 'Max iterations: %d' % maxiter
        if _seed:
            print >> stderr, 'Random seed: %s' % _seed
        print >> stderr, ''

        print >> stderr, 'load word vectors...'
        word_vectors = WordVectors.load_vectors(word_vector_file)
        embsize = word_vectors.embsize()

        print >> stderr, 'init. RAE parameters...'
        timer = Timer()
        timer.tic()
        if _seed != None:
            _seed = int(_seed)
        else:
            _seed = None
        print >> stderr, 'seed: %s' % str(_seed)

        offset = len(word_vectors) * embsize
        theta0 = init_theta(embsize, len(word_vectors), _seed=_seed)
        theta0_init_time = timer.toc()
        print >> stderr, 'shape of theta0 %s' % theta0.shape
        timer.tic()
        if save_theta0:
            print >> stderr, 'saving theta0...'
            pos = model.rfind('.')
            if pos < 0:
                filename = model + '.theta0'
            else:
                filename = model[0:pos] + '.theta0' + model[pos:]
            with Writer(filename) as theta0_writer:
                pickle.dump(theta0, theta0_writer)
        theta0_saving_time = timer.toc()

        print >> stderr, 'preparing data...'
        instances, _, total_internal_node = prepare_rae_data(word_vectors, instances_files)
        print >> stderr, 'amount of instance: %d' % len(instances)
        print >> stderr, 'optimizing...'
        callback = ThetaSaver(model, every)
        func = preTrain
        args = (instances, total_internal_node, word_vectors, embsize, lambda_rec, lambda_reg)
        theta_opt = None
        try:
            theta_opt = lbfgs.optimize(func, theta0[offset:4 * embsize * embsize + 3 * embsize], maxiter, verbose,
                                       checking_grad,
                                       args, callback=callback)
        except GridentCheckingFailedError:
            send_terminate_signal()
            print >> stderr, 'Gradient checking failed, exit'
            exit(-1)

        send_terminate_signal()
        opt_time = timer.toc()

        timer.tic()

        print >> stderr, 'Prepare data...'
        instances, _ = prepare_data(word_vectors, instances_files)
        func = compute_cost_and_grad
        args = (instances, word_vectors, embsize, total_internal_node, lambda_rec,lambda_reg, lambda_reo, instances_of_News, is_Test)
        try:
            print >> stderr, 'Start training...'
            theta_opt = lbfgs.optimize(func, theta0, maxiter, verbose, checking_grad,
                                       args, callback=callback)
        except GridentCheckingFailedError:
            send_terminate_signal()
            print >> stderr, 'Gradient checking failed, exit'
            exit(-1)

        send_terminate_signal()
        opt_time = timer.toc()

        timer.tic()
        # pickle form
        print >> stderr, 'saving parameters to %s' % model
        with Writer(model) as model_pickler:
            pickle.dump(theta_opt, model_pickler)
        # pure text form
        with Writer(model + '.txt') as writer:
            [writer.write('%20.8f\n' % v) for v in theta_opt]
        thetaopt_saving_time = timer.toc()

        print >> stderr, 'Init. theta0  : %10.2f s' % theta0_init_time
        if save_theta0:
            print >> stderr, 'Saving theta0 : %10.2f s' % theta0_saving_time
        print >> stderr, 'Optimizing    : %10.2f s' % opt_time
        print >> stderr, 'Saving theta  : %10.2f s' % thetaopt_saving_time
        print >> stderr, 'Done!'

        print >> stderr, 'Start testing...'

        instances, _ = prepare_test_data(word_vectors, instances_of_News)
        test(instances, theta_opt, word_vectors, isPrint=False)
    else:
        # prepare training data
        instances, word_vectors, total_internal_node = prepare_rae_data()
        embsize = word_vectors.embsize()
        param_size = embsize * embsize * 4 + embsize * 3 + 2 * embsize * 2 + 2
        theta = zeros((param_size, 1))
        offset = embsize * len(word_vectors)
        preTrain(theta[offset:4 * embsize * embsize + 3 * embsize], instances, total_internal_node,
                 word_vectors, embsize, lambda_rec, lambda_reg)
        instances, word_vectors = prepare_data()
        compute_cost_and_grad(theta, instances, word_vectors, embsize, total_internal_node,
                              lambda_rec, lambda_reg, lambda_reo,
                              instances_of_News, is_Test)
