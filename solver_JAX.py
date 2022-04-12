from collections import defaultdict

import csv
import os
import time
import datetime
import numpy as np
import pandas as pd

import pennylane as qml
import random

import torch
import torch.nn.functional as F
import datetime
from utils.utils import *
from models.models import Generator, Discriminator
from models.q_generator_JAX import PatchQuantumGenerator
from data.sparse_molecular_dataset import SparseMolecularDataset
from utils.logger import Logger

from q_discriminator import HybridModel

from frechetdist import frdist

def upper(m, a):
    res = torch.zeros((m.shape[0], 36, 5)).to(m.device).long()
    for i in range(m.shape[0]):
        for j in range(5):
            tmp_m = m[i, :, :, j]
            idx = torch.triu_indices(9, 9,offset = 1)

            res[i, :, j] = tmp_m[list(idx)]
    res = torch.cat((res, a), dim=1)
    return res        

def wasserstein_loss(y_true, y_pred):
	return torch.mean(y_true * y_pred)

class Solver(object):
    """Solver for training and testing MolGAN"""

    def __init__(self, config, log=None):
        """Initialize configurations"""

        # Log
        self.log = log
        
        # loss
        self.wasserstein = True # True only for C dis
        # Data loader
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Quantum
        self.quantum = config.quantum
        self.layer = config.layer
        self.qubits = config.qubits
        self.gen_circuit = config.gen_circuit
        self.update_qc = config.update_qc
        self.qc_lr = config.qc_lr
        self.qc_pretrained = config.qc_pretrained

        # Model configurations
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.la = config.lambda_wgan
        self.la_gp = config.lambda_gp
        self.post_method = config.post_method

        # RL reward suggested by medicinal chemist
        self.metric = 'sas,qed,unique'

        # Training configurations
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        # number of steps per epoch
        self.num_steps =  (len(self.data) // self.batch_size)
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        # learning rate decay
        self.gamma = config.gamma
        self.decay_every_epoch = config.decay_every_epoch

        # critic
        if self.la > 0:
            self.n_critic = config.n_critic
        else:
            self.n_critic = 1
        self.critic_type = config.critic_type

        # Training or test
        self.mode = config.mode
        self.resume_epoch = config.resume_epoch

        # Testing configurations
        self.test_epoch = config.test_epoch
        self.test_sample_size = config.test_sample_size

        # Tensorboard
        self.use_tensorboard = config.use_tensorboard
        if self.mode == 'train' and config.use_tensorboard:
            self.logger = Logger(config.log_dir_path)

        # GPU
        self.device = torch.device("cpu")#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device, flush = True)

        # Directories
        self.log_dir_path = config.log_dir_path
        self.model_dir_path = config.model_dir_path
        self.img_dir_path = config.img_dir_path

        # Step size to save the model
        self.model_save_step = config.model_save_step

        # Build the model
        self.build_model()

        # Quantum
        # quantum or not
        if config.quantum:

            # use pretrained weights or not
            if config.qc_pretrained:
                self.pretrained_qc_weights = pd.read_csv('results/quantum_circuit/molgan_red_weights.csv', header=None).iloc[-1, 1:].values
                self.gen_weights = torch.tensor(list(self.pretrained_qc_weights), requires_grad=True)
            else:
                self.gen_weights = torch.tensor(list(np.random.rand(config.layer*(config.qubits*2-1))*2*np.pi-np.pi), requires_grad=True)

            # learning rate of quantum circuit
            # the learning rate of quantum circuit is different from the learning rate of generator
            if self.update_qc:
                if self.qc_lr:
                    # can use either torch.optim.Adam or torch.optim.RMSprop
                    self.g_optimizer = torch.optim.RMSprop([
                        {'params':list(self.G.parameters())},
                        {'params': [self.gen_weights], 'lr': self.qc_lr}
                    ], lr=self.g_lr)
                else:
                    # can use either torch.optim.Adam or torch.optim.RMSprop
                    self.g_optimizer = torch.optim.RMSprop(list(self.G.parameters())+[self.gen_weights], self.g_lr)
            else:
                # can use either torch.optim.Adam or torch.optim.RMSprop
                self.g_optimizer = torch.optim.RMSprop(list(self.G.parameters()), self.g_lr)

    def build_model(self):
        """Create a generator, a discriminator and a v net"""

        # Models

        self.G = PatchQuantumGenerator(n_generators=45)
        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim - 1, self.dropout)


        # Optimizers can be RMSprop or Adam
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), self.g_lr)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), self.d_lr)

        # Print the networks
        self.print_network(self.G, 'G', self.log)
        self.print_network(self.D, 'D', self.log)
        # self.print_network(self.V, 'V', self.log)
        print("in JAX solver")

        # Bring the network to GPU
        self.G.to(self.device)
        self.D.to(self.device)
        # self.V.to(self.device)

    @staticmethod
    def print_network(model, name, log=None):
        """Print out the network information"""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        if log is not None:
            log.info(model)
            log.info(name)
            log.info("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator"""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_dir_path, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_dir_path, '{}-D.ckpt'.format(resume_iters))
        # V_path = os.path.join(self.model_dir_path, '{}-V.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        # self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def load_gen_weights(self, resume_iters):
        """Restore the trained quantum circuit"""
        weights_pth = os.path.join(self.model_dir_path, 'molgan_red_weights.csv')
        weights = pd.read_csv(weights_pth, header=None).iloc[resume_iters-1, 1:].values
        self.gen_weights = torch.tensor(list(weights), requires_grad=True)

    def update_lr(self, gamma):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] *= gamma
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] *= gamma

    def reset_grad(self):
        """Reset the gradient buffers"""
        self.g_optimizer.zero_grad()
        # self.d_optimizer.zero_grad()
        # self.v_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y, inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors"""
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        """Sample the random noise"""
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess(inputs, method, temperature=1.0):
        """Convert the probability matrices into label matrices"""
        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]
        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))/temperature, hard=False).view(e_logits.size()) for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))/temperature, hard=True).view(e_logits.size()) for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits/temperature, -1) for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def reward(self, mols):
        """Calculate the rewards of mols"""
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.metric == 'all' else self.metric).split(','):
            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))
        return rr.reshape(-1, 1)

    def train_and_validate(self):
        """Train and validate function"""
        self.start_time = time.time()

        # start training from scratch or resume training
        start_epoch = 0
        if self.resume_epoch is not None and self.mode == 'train':
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)
            if self.quantum:
                self.load_gen_weights(self.resume_epoch)
        # restore models for test
        elif self.test_epoch is not None and self.mode == 'test':
            self.restore_model(self.test_epoch)
            if self.quantum:
                self.load_gen_weights(self.test_epoch)
        else:
            print('Training From Scratch...')

        # start training loop or test phase
        if self.mode == 'train':
            print('Start training...')
            for i in range(start_epoch, self.num_epochs):
                self.train_or_valid(epoch_i=i, train_val_test='train')
                self.train_or_valid(epoch_i=i, train_val_test='val')
        elif self.mode == 'test':
            print('Start testing...')
            assert (self.resume_epoch is not None or self.test_epoch is not None)
            self.train_or_valid(epoch_i=start_epoch, train_val_test='val')
        else:
            raise NotImplementedError

    def get_gen_mols(self, n_hat, e_hat, method):
        """Convert edges and nodes matrices into molecules"""
        (edges_hard, nodes_hard) = self.postprocess((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols

    def get_gen_mols_hat(self, n_hat, e_hat, method):
        """Convert edges and nodes matrices into molecules"""
        (edges_hard, nodes_hard) = e_hat, n_hat
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols
    
    def get_reward(self, n_hat, e_hat, method):
        """Get the reward from edges and nodes matrices"""
        (edges_hard, nodes_hard) = self.postprocess((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) for e_, n_ in zip(edges_hard, nodes_hard)]
        reward = torch.from_numpy(self.reward(mols)).to(self.device)
        return reward

    def save_checkpoints(self, epoch_i):
        """store the models and quantum circuit"""
        G_path = os.path.join(self.model_dir_path, '{}-G.ckpt'.format(epoch_i + 1))
        D_path = os.path.join(self.model_dir_path, '{}-D.ckpt'.format(epoch_i + 1))
        # V_path = os.path.join(self.model_dir_path, '{}-V.ckpt'.format(epoch_i + 1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        # torch.save(self.V.state_dict(), V_path)
        # save quantum weights
        if self.quantum:
            with open(os.path.join(self.model_dir_path, 'molgan_red_weights.csv'), 'a') as file:
                writer = csv.writer(file)
                writer.writerow([str(epoch_i)]+list(self.gen_weights.detach().numpy()))
        print('Saved model checkpoints into {}...'.format(self.model_dir_path))
        if self.log is not None:
            self.log.info('Saved model checkpoints into {}...'.format(self.model_dir_path))

    def train_or_valid(self, epoch_i, train_val_test='val'):
        """Train or valid function"""
        # The first several epochs using RL to purse stability (not used)
        if epoch_i < 0:
            cur_la = 0
        else:
            cur_la = self.la

        # Recordings
        losses = defaultdict(list)
        scores = defaultdict(list)

        # Iterations
        the_step = self.num_steps
        if train_val_test == 'val':
            if self.mode == 'train':
                the_step = 1
                print('[Validating]')
            elif self.mode == 'test':
                the_step = 1
                print('[Testing]')
            else:
                raise NotImplementedError
        d_loss = torch.nn.L1Loss() #modified for wasserstein
        for a_step in range(the_step):

            # non-Quantum part
            if train_val_test == 'val' and not self.quantum:
                if self.test_sample_size is None:
                    mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch(1280*4)
                    z = self.sample_z(a.shape[0])
                else:
                    mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch(self.test_sample_size)
                    z = self.sample_z(self.test_sample_size)
            elif train_val_test == 'train' and not self.quantum:
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
                z = self.sample_z(self.batch_size)

            # Quantum part
            elif train_val_test == 'val' and self.quantum:
                if self.test_sample_size is None:
                    mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                    sample_list = [self.gen_circuit(self.gen_weights) for i in range(a.shape[0])]
                else:
                    mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch(self.test_sample_size)
                    sample_list = [self.gen_circuit(self.gen_weights) for i in range(self.test_sample_size)]
            elif train_val_test == 'train' and self.quantum:
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
                sample_list = [self.gen_circuit(self.gen_weights) for i in range(self.batch_size)]

            # Error
            else:
                raise NotImplementedError

            ########## Preprocess input data ##########
            a = torch.from_numpy(a).to(self.device).long() # adjacency
            x = torch.from_numpy(x).to(self.device).long() # node
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)
            
            ax_tensor = upper(a_tensor, x_tensor)

            if self.quantum:
                z = torch.stack(tuple(sample_list)).to(self.device).float()
            else:
                z = torch.from_numpy(z).to(self.device).float()

            # tensorboard
            loss_tb = {}

            # current steps
            cur_step = self.num_steps * epoch_i + a_step
            '''
            ########## Train the discriminator ##########
        
            # compute loss with real inputs
            logits_real, features_real = self.D(a_tensor, None, x_tensor) #// C dis

            target_real = (torch.ones((ax_tensor.shape[0],  1))*-1).to(self.device).long() if self.wasserstein else torch.ones((ax_tensor.shape[0],  1)).to(self.device).long()
            
            # quantum generator
            edges_hat, nodes_hat, edges_hat_GT, nodes_hat_GT = self.G(z)
            # end of quantum generator

            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat) # // C dis      
            target_fake = torch.ones((ax_tensor.shape[0],  1)).to(self.device).long() if self.wasserstein else torch.zeros((ax_tensor.shape[0],  1)).to(self.device).long()

            d_loss_real = d_loss(logits_real, target_real)
            d_loss_fake = d_loss(logits_fake, target_fake)
            loss_D = torch.mean(d_loss_fake) - torch.mean(d_loss_real) if self.wasserstein else (d_loss_real + d_loss_fake) #modified for wasserstein

            if cur_la > 0:
                losses['D/loss_real'].append(d_loss_real.item())
                losses['D/loss_fake'].append(d_loss_fake.item())
                #losses['D/loss_gp'].append(grad_penalty.item())
                losses['D/loss'].append(loss_D.item())

                # tensorboard
                loss_tb['D/loss_real'] = d_loss_real.item()
                loss_tb['D/loss_fake'] = d_loss_fake.item()
                #loss_tb['D/loss_gp'] = grad_penalty.item()
                loss_tb['D/loss'] = loss_D.item()

            # Optimise discriminator
            if train_val_test == 'train':
                if self.critic_type == 'D':
                    # training D for n_critic-1 times followed by G one time
                    if (cur_step == 0) or (cur_step % self.n_critic != 0):
                        self.reset_grad()
                        loss_D.backward()
                        self.d_optimizer.step()
                        
                        ### here is to check discriminator gradient value ###
                        for i, param in enumerate(self.D.parameters()):
                            if self.wasserstein:
                                param.data.clamp_(-0.01, 0.01)
                            else:
                                print(param.grad)
                                break
                        print('optimizing D')
                        #####################################################
                        
                        
                        for name, param in reversed(list(self.G.named_parameters())):
                            if param.requires_grad:
                                print (name, param.grad)
                            break
                        print("'optimizing G")
                        
                else:
                    # training G for n_critic-1 times followed by D one time
                    if (cur_step != 0) and (cur_step % self.n_critic == 0):
                        self.reset_grad()
                        loss_D.backward()
                        self.d_optimizer.step()
                        print('-') 
            ########## Train the generator ##########
            '''
            # Z-to-target

            # quantum generator
            edges_hat, nodes_hat, edges_hat_GT, nodes_hat_GT = self.G(z)
            # end of quantum generator
            # logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)   #// C dis
            
            # target_fake = torch.ones((ax_tensor.shape[0],  1)).to(self.device).long() if self.wasserstein else torch.zeros((ax_tensor.shape[0],  1)).to(self.device).long()


            # Losses Update
            # loss_G = -1* d_loss(logits_fake, target_fake)#-logits_fake #modified for wasserstein
            loss_G = 1* (d_loss(edges_hat, edges_hat_GT) + d_loss(nodes_hat, nodes_hat_GT))
            # print(edges_hat[0], edges_hat_GT[1], edges_hat.shape, edges_hat_GT.shape)
            # print(nodes_hat[0], nodes_hat_GT[1], nodes_hat.shape, nodes_hat_GT.shape)
            # print(torch.sum(torch.abs(edges_hat-edges_hat_GT)))
            # print(torch.sum(torch.abs(nodes_hat-nodes_hat_GT)))
            loss_G = torch.mean(loss_G)
            losses['G/loss'].append(loss_G.item())


            # tensorboard
            loss_tb['G/loss'] = loss_G.item()

            # print('d_loss {:.2f} d_fake {:.2f} d_real {:.2f} g_loss: {:.2f}'.format(loss_D.item(), d_loss_fake.item(), d_loss_real.item(), loss_G.item()))
            print('g_loss: {:.2f}'.format(loss_G.item()))
            print('======================= {} =============================='.format(datetime.datetime.now()), flush = True)
            # alpha = torch.abs(loss_G.detach() / loss_RL.detach()).detach()
            train_step_G = cur_la * loss_G# + (1.0 - cur_la) * alpha * loss_RL

            # train_step_V = loss_V
            self.reset_grad()
            train_step_G.backward(retain_graph=True)
            self.g_optimizer.step()
            # Optimise generator and reward network
            # if train_val_test == 'train':
            #     if self.critic_type == 'D':
            #         # training D for n_critic-1 times followed by G one time
            #         if (cur_step != 0) and (cur_step % self.n_critic) == 0:
            #             self.reset_grad()
            #             if cur_la < 1.0:
            #                 train_step_G.backward(retain_graph=True)
            #                 # train_step_V.backward()
            #                 self.g_optimizer.step()
            #                 # self.v_optimizer.step()
            #             else:
            #                 train_step_G.backward(retain_graph=True)
            #                 self.g_optimizer.step()
            #     else:
            #         # training G for n_critic-1 times followed by D one time
            #         if (cur_step == 0) or (cur_step % self.n_critic != 0):
            #             self.reset_grad()
            #             if cur_la < 1.0:
            #                 train_step_G.backward(retain_graph=True)
            #                 # train_step_V.backward()
            #                 self.g_optimizer.step()
            #                 # self.v_optimizer.step()
            #             else:
            #                 train_step_G.backward(retain_graph=True)
            #                 self.g_optimizer.step()


            if train_val_test == 'train' and self.use_tensorboard:
                for tag, value in loss_tb.items():
                    self.logger.scalar_summary(tag, value, cur_step)


            ########## Frechet distribution ##########
            
            (edges_hard, nodes_hard) = edges_hat, nodes_hat#self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            R = [list(a[i].reshape(-1).to('cpu'))  for i in range(self.batch_size)]
            F = [list(edges_hard[i].reshape(-1).to('cpu'))  for i in range(self.batch_size)]
            #F =  F.cpu()
            fd_bond = frdist(R, F)

            R=[list(x[i].to('cpu')) + list(a[i].reshape(-1).to('cpu'))  for i in range(self.batch_size)]
            F=[list(nodes_hard[i].to('cpu')) + list(edges_hard[i].reshape(-1).to('cpu'))  for i in range(self.batch_size)]
            fd_bond_atom = frdist(R, F)

            loss_tb['FD/bond'] = fd_bond
            loss_tb['FD/bond_atom'] = fd_bond_atom

            losses['FD/bond'].append(fd_bond)
            losses['FD/bond_atom'].append(fd_bond_atom)


            if train_val_test == 'train' and self.use_tensorboard:
                for tag, value in loss_tb.items():
                    self.logger.scalar_summary(tag, value, cur_step)

            ########## Miscellaneous ##########

            # Decay learning rates
            if epoch_i != 0 and self.decay_every_epoch:
                if a_step == 0 and (epoch_i+1) % self.decay_every_epoch == 0:
                    self.update_lr(self.gamma)


            # Get scores
            if train_val_test == 'val':
                mols = self.get_gen_mols_hat(nodes_hat, edges_hat, self.post_method)
                m0, m1 = all_scores(mols, self.data, norm=True)  # 'mols' is output of Fake Reward
                for k, v in m1.items():
                    scores[k].append(v)
                for k, v in m0.items():
                    scores[k].append(np.array(v)[np.nonzero(v)].mean())

                # Save checkpoints
                if self.mode == 'train':
                    if (epoch_i + 1) % self.model_save_step == 0:
                        self.save_checkpoints(epoch_i=epoch_i)

                # Saving molecule images
                mol_f_name = os.path.join(self.img_dir_path, 'mol-{}.png'.format(epoch_i))
                save_mol_img(mols, mol_f_name, is_test=self.mode == 'test')

                # Print out training information
                et = time.time() - self.start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]:".format(et, epoch_i + 1, self.num_epochs)

                is_first = True
                for tag, value in losses.items():
                    if is_first:
                        log += "\n{}: {:.2f}".format(tag, np.mean(value))
                        is_first = False
                    else:
                        log += ", {}: {:.2f}".format(tag, np.mean(value))
                is_first = True
                for tag, value in scores.items():
                    if is_first:
                        log += "\n{}: {:.2f}".format(tag, np.mean(value))
                        is_first = False
                    else:
                        log += ", {}: {:.2f}".format(tag, np.mean(value))
                print(log)


                if self.log is not None:
                    self.log.info(log)
