mod seq;
use crate::model::seq::seq;
use candle_core::{DType, Device, Error, Module, Result, Tensor};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::{
    conv2d, init, linear, Activation, AdamW, Conv2dConfig, Optimizer, VarBuilder, VarMap,
};
use num_traits::ToPrimitive;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use seq::Sequential;
use std::fmt::{Debug, Formatter};
use tracing::{debug, info, trace};
#[derive(Clone)]
pub struct Step {
    pub input: Tensor,
    pub reward: f32,
    pub action: i64,
    pub terminated: bool,
}
impl Debug for Step {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Step")
            .field("reward", &self.reward)
            .field("action", &self.action)
            .field("terminated", &self.terminated)
            .finish()
    }
}

fn weighted_sample(probs: Vec<f32>, rng: &mut ThreadRng) -> Result<usize> {
    let distribution = rand::distributions::WeightedIndex::new(probs).map_err(Error::wrap)?;
    let mut rng = rng;
    Ok(distribution.sample(&mut rng))
}

pub struct LearnOutput {
    pub loss: f32,
    pub accuracy: f32,
}

pub struct Model {
    nn: Sequential,
    space: usize,
    action_space: usize,
}

impl Model {
    pub fn new(
        varmap: &VarMap,
        device: &Device,
        space: usize,
        action_space: usize,
    ) -> Result<Model> {
        device.synchronize()?;
        let vb = VarBuilder::from_varmap(varmap, DType::F32, &device);
        let init_ws = init::DEFAULT_KAIMING_NORMAL;

        let model = seq()
            .add(conv2d(4, 64, 4, Conv2dConfig::default(), vb.pp("conv2d"))?)
            .add_fn(|a| a.flatten_from(1))
            .add(linear(256, 64, vb.pp("linear_in"))?)
            .add(Activation::Relu)
            .add(linear(64, action_space, vb.pp("linear_out"))?);

        Ok(Model {
            nn: model,
            space,
            action_space,
        })
    }

    pub fn predict(&self, state: &Tensor, rng: &mut ThreadRng) -> Result<usize> {
        let action = {
            let logits = self
                .nn
                .forward(&state.detach().unsqueeze(0)?)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap();
            let action_probs: Vec<f32> = softmax(&logits, 0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            // let select: Vec<u32> = action_probs
            //     .unsqueeze(0)?
            //     .argmax(1)
            //     .unwrap()
            //     .to_vec1()
            //     .unwrap();
            let select = weighted_sample(action_probs.clone(), rng)? as i64;

            trace!(
                "Action Probability {:?} Selected {:?}",
                action_probs,
                select
            );
            select
        };

        return Ok(action.to_usize().unwrap());
    }
    pub fn learn(&mut self, steps: Vec<Step>, device: &Device, opt: &mut AdamW) -> Result<()> {
        let rewards = Tensor::from_vec(accumulate_rewards(&steps), steps.len(), device)?
            .to_dtype(DType::F32)?
            .detach();
        debug!(
            "Accumulated Rewards: {:?}",
            rewards.to_vec1::<f32>().unwrap()[rewards.shape().dims1().unwrap() - 1]
        );
        let actions_mask = {
            let actions: Vec<i64> = steps.iter().map(|s| s.action).collect();
            let actions_mask: Vec<Tensor> = actions
                .iter()
                .map(|&action| {
                    // One-hot encoding
                    let mut action_mask: Vec<f32> = vec![0.0; self.action_space];
                    action_mask[action as usize] = 1.0;

                    Tensor::from_vec(action_mask, self.action_space, &device).unwrap()
                })
                .collect();
            Tensor::stack(&actions_mask, 0)?.detach()
        };

        let states = {
            let states: Vec<Tensor> = steps.into_iter().map(|s| s.input).collect();
            Tensor::stack(&states, 0)?.detach()
        };

        let log_probs = actions_mask
            .mul(&log_softmax(&self.nn.forward(&states).unwrap().squeeze(1).unwrap(), 1).unwrap())
            .unwrap()
            .sum(1)?;

        let loss = rewards.mul(&log_probs)?.neg()?.mean_all()?;
        debug!(
            "Loss: {:?} On {:?}",
            loss.to_scalar::<f32>()?,
            states.shape()
        );
        opt.backward_step(&loss)?;

        Ok(())
    }
}

fn accumulate_rewards(steps: &[Step]) -> Vec<f32> {
    let mut rewards: Vec<f32> = steps.iter().map(|s| s.reward).collect();
    let mut acc_reward = 0f32;
    for (i, reward) in rewards.iter_mut().enumerate().rev() {
        if steps[i].terminated {
            acc_reward = 0.0;
        }
        acc_reward += *reward;
        *reward = acc_reward;
    }
    rewards
}
