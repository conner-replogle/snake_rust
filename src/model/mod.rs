mod seq;
use crate::game::GameState;
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
    pub input: (Tensor, Tensor),
    pub reward: f32,
    pub action: i64,
    pub terminated: bool,
    pub state: GameState,
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
    pub step_per_games: f32,
    pub max_score: u32,
    pub avg_score: f32,
    pub death_by_wall: u32,
    pub death_by_self: u32,
    pub death_wasted_moves: u32,
    pub wons: u32,
}

fn global_max_pool2d(tensor: &Tensor) -> Result<Tensor> {
    // Get the shape of the input tensor
    let shape = tensor.shape().dims4()?;

    // Assume input tensor is in the format (batch_size, channels, height, width)
    let height = shape.2; // Get height of the feature map
    let width = shape.3; // Get width of the feature map

    // Apply max pooling with kernel size equal to the entire height and width of the feature map
    let pooled_tensor = tensor.max_pool2d((height, width))?;
    // The result will have spatial dimensions of (1, 1), reducing it to (batch_size, channels, 1, 1)
    Ok(pooled_tensor)
}

pub struct NerualNet {
    conv_net: Sequential,
    num_net: Sequential,
    out_net: Sequential,
}
impl NerualNet {
    pub fn new(varmap: &VarMap, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, &device);
        let out_c = 64;
        let k = 3;
        let conv_seq = seq()
            .add(conv2d(
                3,
                out_c,
                k,
                Conv2dConfig::default(),
                vb.pp("conv2d"),
            )?)
            .add_fn(|a| global_max_pool2d(a)?.flatten_from(1)); // Global pooling to handle variable input sizes
        let num_seq = seq()
            .add(linear(12, 64, vb.pp("num_linear1"))?)
            .add(Activation::Relu)
            .add(linear(64, 64, vb.pp("num_linear2"))?)
            .add(Activation::Relu);

        let output_seq = seq()
            .add(linear(out_c + 64, 128, vb.pp("out_linear1"))?)
            .add(Activation::Relu)
            .add(linear(128, 4, vb.pp("out_linear2"))?);
        return Ok(NerualNet {
            conv_net: conv_seq,
            num_net: num_seq,
            out_net: output_seq,
        });
    }
    pub fn forward(&self, input: &(Tensor, Tensor)) -> Result<Tensor> {
        let input = if input.0.shape().dims().len() == 3 {
            (input.0.unsqueeze(0)?, input.1.unsqueeze(0)?)
        } else {
            input.clone()
        };
        let conv = self.conv_net.forward(&input.0)?;
        let num = self.num_net.forward(&input.1)?;
        let out = self.out_net.forward(&Tensor::cat(&[conv, num], 1)?);
        return out;
    }
}

pub struct Model {
    nn: NerualNet,
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
        let nn = NerualNet::new(varmap, device)?;
        Ok(Model {
            nn,
            space,
            action_space,
        })
    }

    pub fn predict(&self, state: &(Tensor, Tensor), rng: &mut ThreadRng) -> Result<usize> {
        let action = {
            let logits = self.nn.forward(state).unwrap().squeeze(0).unwrap();
            let action_probs = softmax(&logits, 0).unwrap().squeeze(0).unwrap();

            let select: u32 = action_probs.argmax(0).unwrap().to_scalar().unwrap();
            // let select =
            // weighted_sample(action_probs.clone().to_vec1::<f32>().unwrap(), rng)? as u32;
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
        let (moves, deaths, games, rewards) = accumulate_rewards(&steps);
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

        let states: (Tensor, Tensor) = {
            let (a, b): (Vec<Tensor>, Vec<Tensor>) = steps.into_iter().map(|s| s.input).unzip();
            (
                Tensor::stack(&a, 0)?.detach(),
                Tensor::stack(&b, 0)?.detach(),
            )
        };

        let log_probs = actions_mask
            .mul(&log_softmax(&self.nn.forward(&states).unwrap().squeeze(1).unwrap(), 1).unwrap())
            .unwrap()
            .sum(1)?;
        let log_probs = log_probs.clamp(-1e10, 1e10)?; // Prevent overly large log values

        let loss = rewards.mul(&log_probs)?.neg()?.mean_all()?;

        opt.backward_step(&loss)?;
        let steps = states.1.shape().dims()[0];
        Ok(LearnOutput {
            time: 0,
            left: moves[0],
            right: moves[1],
            up: moves[2],
            down: moves[3],
            step_per_games: steps as f32 / games.len() as f32,

            loss: loss.to_scalar().unwrap(),
            highest_reward: highest_reward_game,
            average_reward: average_reward_game,
            steps,
            games: games.len(),
            max_score: games.iter().cloned().max().unwrap(),
            avg_score: games.iter().cloned().sum::<u32>() as f32 / games.len() as f32,
            death_by_wall: deaths[0],
            death_by_self: deaths[1],
            death_wasted_moves: deaths[2],
            wons: deaths[3],
        })
    }
}

pub fn accumulate_rewards(steps: &[Step]) -> ([u32; 4], [u32; 4], Vec<u32>, Vec<f32>) {
    let mut rewards: Vec<f32> = steps.iter().map(|s| s.reward).collect();
    let mut acc_reward = 0f32;
    let mut games = vec![];
    let mut moves = [0, 0, 0, 0];
    let mut deaths = [0, 0, 0, 0];
    let mut score = 0;

    for (i, reward) in rewards.iter_mut().enumerate().rev() {
        if steps[i].terminated {
            acc_reward = 0.0;
            match steps[i].state {
                GameState::DiedByWall => deaths[0] += 1,
                GameState::DiedBySelf => deaths[1] += 1,
                GameState::WastedMoves => deaths[2] += 1,
                GameState::Won => deaths[3] += 1,
                _ => {}
            }

            games.push(score);
            score = 0;
        }
        if steps[i].state == GameState::AteFood {
            score += 1;
        }
        moves[steps[i].action as usize] += 1;
        acc_reward += *reward;
        *reward = acc_reward;
    }
    (moves, deaths, games, rewards)
}
