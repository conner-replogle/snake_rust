mod seq;
use crate::model::seq::seq;
use candle_core::{DType, Device, Error, Module, Result, Tensor};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::{
    conv2d, init, linear, Activation, AdamW, Conv2dConfig, Optimizer, VarBuilder, VarMap,
};
use chrono::{DateTime, Utc};
use num_traits::ToPrimitive;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use seq::Sequential;
use serde::Serialize;
use std::fmt::{Debug, Formatter};
use std::time::Instant;
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
#[derive(Debug, Serialize)]
pub struct LearnOutput {
    pub time: usize,
    pub loss: f32,
    pub left: u32,
    pub right: u32,
    pub up: u32,
    pub down: u32,
    pub highest_reward: f32,
    pub average_reward: f32,
    pub steps: usize,
    pub games: usize,
    pub step_per_games: u32,
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
            let action_probs = softmax(&logits, 0).unwrap().squeeze(0).unwrap();

            // let select: u32 = action_probs.argmax(0).unwrap().to_scalar().unwrap();
            let select =
                weighted_sample(action_probs.clone().to_vec1::<f32>().unwrap(), rng)? as u32;
            assert!(select < self.action_space.to_u32().unwrap());
            trace!(
                "Action Probability {:?} Selected {:?}",
                action_probs,
                select
            );
            select
        };

        return Ok(action.to_usize().unwrap());
    }
    pub fn learn(
        &mut self,
        steps: Vec<Step>,
        device: &Device,
        opt: &mut AdamW,
    ) -> Result<LearnOutput> {
        let (moves, games, rewards) = accumulate_rewards(&steps);
        let rewards = Tensor::from_vec(rewards, steps.len(), device)?
            .to_dtype(DType::F32)?
            .detach();
        let highest_reward_game = rewards
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .reduce(f32::max)
            .unwrap();
        let average_reward_game =
            rewards.to_vec1::<f32>()?.iter().sum::<f32>() / steps.len() as f32;

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
        let log_probs = log_probs.clamp(-1e10, 1e10)?; // Prevent overly large log values

        let loss = rewards.mul(&log_probs)?.neg()?.mean_all()?;

        opt.backward_step(&loss)?;
        let steps = states.shape().dims()[0];
        Ok(LearnOutput {
            time: 0,
            left: moves[0],
            right: moves[1],
            up: moves[2],
            down: moves[3],
            step_per_games: steps as u32 / games as u32,

            loss: loss.to_scalar().unwrap(),
            highest_reward: highest_reward_game,
            average_reward: average_reward_game,
            steps,
            games,
        })
    }
}

fn accumulate_rewards(steps: &[Step]) -> ([u32; 4], usize, Vec<f32>) {
    let mut rewards: Vec<f32> = steps.iter().map(|s| s.reward).collect();
    let mut acc_reward = 0f32;
    let mut games = 0;
    let mut moves = [0, 0, 0, 0];
    for (i, reward) in rewards.iter_mut().enumerate().rev() {
        if steps[i].terminated {
            acc_reward = 0.0;
            games += 1;
        }
        moves[steps[i].action as usize] += 1;
        acc_reward += *reward;
        *reward = acc_reward;
    }
    (moves, games, rewards)
}
