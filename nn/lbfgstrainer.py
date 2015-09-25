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
             word_vectors, embsize, lambda_reg):
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

        #send data
        instance_num = len(instances)
        esize = int(instance_num/worker_num+0.5)
        sizes = [esize] * worker_num
        sizes[-1] = instance_num - esize*(worker_num-1)
        offset = sizes[0]
        for i in range(1, worker_num):
            comm.send(instances[offset:offset+sizes[i]], dest=i)
            offset += sizes[i]
        comm.barrier()
        local_instance_strs = instances[0:sizes[0]]

        # init recursive autoencoder
        rae = RecursiveAutoencoder.build(theta, embsize)

        # compute local reconstruction error and gradients
        rec_error, gradient_vec = process_rae_local_batch(rae, word_vectors, local_instance_strs)

        # compute total reconstruction error
        total_rec_error = comm.reduce(rec_error, op=MPI.SUM, root=0)

        # compute total cost
        reg = rae.get_weights_square()
        total_cost = total_rec_error / total_internal_node_num + lambda_reg / 2 * reg

        # compute gradients
        total_grad = zeros_like(gradient_vec)
        comm.Reduce([gradient_vec, MPI.DOUBLE], [total_grad, MPI.DOUBLE],
                    op=MPI.SUM, root=0)
        total_grad /= total_internal_node_num

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

            # receive data
            local_instance_strs = comm.recv(source=0)
            comm.barrier()

            # init recursive autoencoder
            rae = RecursiveAutoencoder.build(theta, embsize)

            # compute local reconstruction error and gradients
            rec_error, gradient_vec = process_rae_local_batch(rae, word_vectors, local_instance_strs)

            # send local reconstruction error to root
            comm.reduce(rec_error, op=MPI.SUM, root=0)

            # send local gradients to root
            comm.Reduce([gradient_vec, MPI.DOUBLE], None, op=MPI.SUM, root=0)


def compute_cost_and_grad(theta, instances, instances_of_Unlabel, word_vectors, embsize, lambda_reg, lambda_reo):
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

        # init rae
        rae = RecursiveAutoencoder.build(theta, embsize)

        offset = RecursiveAutoencoder.compute_parameter_num(embsize)

        rm = ReorderClassifer.build(theta, embsize, rae)

        # compute local reconstruction error, reo and gradients
        local_error, rae_gradient, rm_gradient = process_local_batch(rm, rae, word_vectors, instances, lambda_reo)

        # compute total reconstruction error
        total_error = comm.reduce(local_error, op=MPI.SUM, root=0)

        # compute total cost
        reg = rae.get_weights_square() + rm.get_weights_square()
        final_cost = total_error / len(instances) + lambda_reg / 2 * reg

        # compute gradients
        total_rae_grad = zeros_like(rae_gradient)
        total_rm_grad = zeros_like(rm_gradient)
        comm.Reduce([rae_gradient, MPI.DOUBLE], [total_rae_grad, MPI.DOUBLE],
                    op=MPI.SUM, root=0)
        comm.Reduce([rm_gradient, MPI.DOUBLE], [total_rm_grad, MPI.DOUBLE],
                    op=MPI.SUM, root=0)
        total_rae_grad /= len(instances)
        total_rm_grad /= len(instances)

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

        return total_error, concatenate((total_rae_grad, total_rm_grad))
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
            offset = RecursiveAutoencoder.compute_parameter_num()
            rm = ReorderClassifer.build(theta[offset:], embsize)

            # compute local reconstruction error, reo and gradients
            local_error, rae_gradient, rm_gradient = process_local_batch(rm, rae, word_vectors, instances, lambda_reo)

            # send local reconstruction error to root
            comm.reduce(local_error, op=MPI.SUM, root=0)

            # send local gradients to root
            comm.Reduce([rae_gradient, MPI.DOUBLE], None, op=MPI.SUM, root=0)
            comm.Reduce([rm_gradient, MPI.DOUBLE], None, op=MPI.SUM, root=0)


def process_rae_local_batch(rae, word_vectors, instances):
    gradients = rae.get_zero_gradients()
    total_rec_error = 0
    for instance in instances:
        words_embedded = word_vectors[instance.words]
        root_node, rec_error = rae.forward(words_embedded)
        rae.backward(root_node, gradients, freq=instance.freq)
        total_rec_error += rec_error * instance.freq

    return total_rec_error, gradients.to_row_vector()


def process_local_batch(rm, rae, word_vectors, instances, lambda_reo):
    rae_gradients = rae.get_zero_gradients()
    rm_gradients = rm.get_zero_gradients()
    total_error = 0
    for instance in instances:
        words_embedded = word_vectors[instance.preWords]
        root_prePhrase, rec_error = rae.forward(words_embedded)
        total_error += rec_error

        words_embedded = word_vectors[instance.aftWords]
        root_aftPhrase, rec_error = rae.forward(words_embedded)
        total_error += rec_error

        softmaxLayer, reo_error = rm.forward(instance, root_prePhrase.p, root_aftPhrase.p, embsize)
        total_error += reo_error * lambda_reo
        delta_to_left, delta_to_right = rm.backward(softmaxLayer, instance.order, root_prePhrase.p, root_aftPhrase.p,
                                                    rm_gradients)
        rae.backward(root_prePhrase, rae_gradients, delta_to_left)
        rae.backward(root_aftPhrase, rae_gradients, delta_to_right)

    return total_error, rae_gradients.to_row_vector(), rm_gradients.to_row_vector()


def init_theta(embsize, num_of_domains=1, _seed=None):
    if _seed != None:
        ori_state = get_state()
        seed(_seed)

    parameters = []

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

    for i in range(0, num_of_domains):
        parameters.append(init_W(1, embsize * 2))
        parameters.append(init_W(1, embsize * 2))
        parameters.append(zeros(1))
        parameters.append(zeros(1))

    if _seed != None:
        set_state(ori_state)

    return concatenate(parameters)


def prepare_rae_data(word_vectors=None, datafile=None, unlabel_file=None):
    '''Prepare training data for rae
    Args:
      word_vectors: an instance of vec.wordvector
      datafile: location of data file

    Return:
      instances: a list of Instance
      word_vectors: word_vectors
      total_internal_node: total number of internal nodes
    '''
    # broadcast word vectors
    instance_strs = []
    # load raw data
    for file in datafile:
        with Reader(file) as file:
            for line in file:
                phrases = getPhrases(line)
                instance_strs.append(phrases[0])
                instance_strs.append(phrases[1])

    for line in unlabel_file:
        phrases = line.split("\t")
        instance_strs.append(phrases[0])
        instance_strs.append(phrases[1])

    instances, total_internal_node = load_rae_instances(instance_strs,
                                                        word_vectors)
    return instances, word_vectors, total_internal_node


def prepare_data(word_vectors=None, dataFile=None, unlabelFile=None):
    '''Prepare training data
    Args:
    word_vectors: an instance of vec.wordvector
    dataFile: raw training file

    Return:
    instances: a list of ReorderInstance
    word_vectors
    '''
    instance_of_domain = []
    instance_lines = []
    lines_of_Unlabel = []
    instances_of_Unlabel = []

    if unlabelFile != None:
        for line in unlabelFile:
            lines_of_Unlabel.append(line)

        instances_of_Unlabel = [ReorderInstance.paser_from_str(i, word_vectors) for i in lines_of_Unlabel]
        instances_of_Unlabel = [i for i in instances_of_Unlabel if len(i.preWords) != 0 and len(i.aftWords) != 0]

    if type(dataFile) == str:
        with Reader(dataFile) as file:
            for line in file:
                instance_of_domain.append(line)
        instances = load_instances(instance_of_domain, word_vectors)
        if unlabelFile != None:
            return instances, instances_of_Unlabel, word_vectors
        return instances, word_vectors

    for file in dataFile:
        with Reader(file) as file:
            for line in file:
                instance_of_domain.append(line)
        instance_lines.append(instance_of_domain)
        instance_of_domain = []

    instances_of_domains = []
    for i in range(0, len(instance_lines)):
        instances = load_instances(instance_lines[i], word_vectors)
        instances_of_domains.append(instances)

    del instance_lines

    if unlabelFile != None:
        return instances_of_domains, instances_of_Unlabel, word_vectors

    return instances_of_domains, word_vectors


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


def test(instances, theta, word_vectors):
    outfile = open('./output/test_result.txt', 'w')
    total_lines = len(instances)
    total_true = 0

    # init rae
    rae = RecursiveAutoencoder.build(theta, embsize)

    offset = RecursiveAutoencoder.compute_parameter_num(embsize)

    rm = ReorderClassifer.build(theta, embsize, rae)

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

        outfile.write("%f\t[%f,%f]\n" % (instance.order, softmaxLayer[0], softmaxLayer[1]))

    outfile.write("Total instances: %f\tTotal true predictions: %f" % (total_lines, total_true))
    outfile.write("Precision: %f" % (float(total_true / total_lines)))


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
    parser.add_argument('-instances_of_Unlabel', required=False)
    parser.add_argument('-isTest', required=False, default=True)
    parser.add_argument('-instances_of_News', required=False)
    parser.add_argument('-model', required=True,
                        help='model name')
    parser.add_argument('-word_vector', required=True,
                        help='word vector file', )
    parser.add_argument('-lambda_reg', type=float, default=0.15,
                        help='weight of the regularizer')
    parser.add_argument('-lambda_reo', type=float, default=0.15,
                        help='weight of the reo')
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
    num_of_domains = len(instances_files)
    instances_file_of_Unlabel = options.instances_of_Unlabel

    model = options.model
    word_vector_file = options.word_vector
    lambda_reg = options.lambda_reg
    lambda_reo = options.lambda_reo
    save_theta0 = options.save_theta0
    checking_grad = options.checking_grad
    maxiter = options.maxiter
    every = options.every
    _seed = options.seed
    verbose = options.verbose
    is_Test = options.isTest
    instances_of_News = options.instances_of_News

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
    print >> stderr, 'Max iterations: %d' % maxiter
    if _seed:
        print >> stderr, 'Random seed: %s' % _seed
    print >> stderr, ''

    print >> stderr, 'load word vectors...'
    word_vectors = WordVectors.load_vectors(word_vector_file)
    embsize = word_vectors.embsize()

    print >> stderr, 'preparing data...'
    instances, _, total_internal_node = prepare_rae_data(word_vectors, instances_files, instances_file_of_Unlabel)

    print >> stderr, 'init. RAE parameters...'
    timer = Timer()
    timer.tic()
    if _seed != None:
        _seed = int(_seed)
    else:
        _seed = None
    print >> stderr, 'seed: %s' % str(_seed)

    theta0 = init_theta(embsize, num_of_domains, _seed=_seed)
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

    print >> stderr, 'optimizing...'

    callback = ThetaSaver(model, every)
    func = preTrain
    args = (instances, total_internal_node, word_vectors, embsize, lambda_reg)
    theta_opt = None
    try:
        theta_opt = lbfgs.optimize(func, theta0[0:4 * embsize * embsize + 3 * embsize], maxiter, verbose,
                                   checking_grad,
                                   args, callback=callback)
    except GridentCheckingFailedError:
        send_terminate_signal()
        print >> stderr, 'Gradient checking failed, exit'
        exit(-1)

    print >> stderr, 'Start training rm...'
    instances, instances_of_Unlabel, _ = prepare_data(word_vectors, instances_files, instances_file_of_Unlabel)
    func = compute_cost_and_grad
    args = (instances, instances_of_Unlabel, word_vectors, embsize, lambda_reg, lambda_reo)
    try:
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

    if is_Test:
        print >> stderr, 'Start testing...'

        instances, _ = prepare_data(word_vectors, instances_of_News)
        test(instances, theta0, word_vectors)
