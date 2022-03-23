import torch
from torch import nn
from torch import distributions as torchd

import models
import networks
import tools


class Random(nn.Module):

  def __init__(self, config):
    self._config = config

  def actor(self, feat):
    shape = feat.shape[:-1] + [self._config.num_actions]
    if self._config.actor_dist == 'onehot':
      return tools.OneHotDist(torch.zeros(shape))
    else:
      ones = torch.ones(shape)
      return tools.ContDist(torchd.uniform.Uniform(-ones, ones))

  def train(self, start, context):
    return None, {}


#class Plan2Explore(tools.Module):
class Plan2Explore(nn.Module):

  def __init__(self, config, world_model, reward=None):
    super(Plan2Explore, self).__init__()
    self._config = config
    self._use_amp = True if config.precision==16 else False
    self._reward = reward
    self._behavior = models.ImagBehavior(config, world_model, prestr="_expl")
    self.actor = self._behavior.actor
    stoch_size = config.dyn_stoch
    if config.dyn_discrete:
      stoch_size *= config.dyn_discrete
    size = {
        'embed': 32 * config.cnn_depth,
        'stoch': stoch_size,
        'deter': config.dyn_deter,
        'feat': config.dyn_stoch + config.dyn_deter,
    }[self._config.disag_target]
    kw = dict(
        inp_dim=config.dyn_stoch + config.dyn_deter,  # pytorch version
        shape=size, layers=config.disag_layers, units=config.disag_units,
        act=config.act)
    self._networks = [
        networks.DenseHead(**kw).to(self._config.device) for _ in range(config.disag_models)]

    kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
    parameters = tools.get_parameters(self._networks)

    self._opt = tools.Optimizer(
        'disag_expl', parameters, config.model_lr, config.opt_eps, config.actor_grad_clip,
        **kw)

    #self._opt = tools.Optimizer(
    #    'ensemble', config.model_lr, config.opt_eps, config.grad_clip,
    #    config.weight_decay, opt=config.opt)

  def train(self, start, context, data, decoder, prestr):
    metrics = {}
    stoch = start['stoch']
    if self._config.dyn_discrete:
      stoch = torch.reshape(
          stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
    target = {
        'embed': context['embed'],
        'stoch': stoch,
        'deter': start['deter'],
        'feat': context['feat'],
    }[self._config.disag_target]
    inputs = context['feat']
    if self._config.disag_action_cond:
      inputs = torch.concat([inputs, data['action']], -1)
    metrics.update(self._train_ensemble(inputs, target))
    metrics.update(self._behavior._train(start, self._intrinsic_reward, 
                                      decoder=decoder, prestr = prestr)[-1])
    return None, metrics

  def _intrinsic_reward(self, feat, state, action):
    inputs = feat
    if self._config.disag_action_cond:
      inputs = torch.concat([inputs, action], -1)
    preds = [head(inputs, torch.float32).mean for head in self._networks]
    preds = torch.stack(preds)
    disag = torch.mean(torch.std(preds, 0), -1)
    if self._config.disag_log:
      disag = torch.log(disag)
    reward = self._config.expl_intr_scale * disag
    if self._config.expl_extr_scale:
      reward += (self._config.expl_extr_scale * self._reward(
          feat, state, action)).type(torch.float32)
    reward = reward.unsqueeze(-1)
    return reward

  def _train_ensemble(self, inputs, targets):
    if self._config.disag_offset:
      targets = targets[:, self._config.disag_offset:]
      inputs = inputs[:, :-self._config.disag_offset]
    targets = targets.detach()
    inputs = inputs.detach()
    parameters = tools.get_parameters(self._networks)
    with tools.RequiresGrad(parameters):
      with torch.cuda.amp.autocast(self._use_amp):
        preds = [head(inputs) for head in self._networks]
        likes = [torch.mean(pred.log_prob(targets)) for pred in preds]
        likes = torch.stack(likes)
        loss = -torch.sum(likes).type(torch.float32)
    # print(preds[0])
    # print(likes[0])
    # print(loss)
    with tools.RequiresGrad(parameters):
      metrics = self._opt(loss, parameters)
    return metrics