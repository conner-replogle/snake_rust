use candle_core::{DType, Device, Error, Module, Result, Tensor};
use candle_nn::loss::mse;
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::{
    init, linear, seq, Activation, AdamW, Optimizer, ParamsAdamW, Sequential, VarBuilder, VarMap,
};
use num_traits::ToPrimitive;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use std::fmt::{Debug, Formatter};
use tracing::{debug, info};
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
        tracing::info!(
            "Creating Model {:?}",
            vb.pp("linear_in")
                .get_with_hints((64, 400), "weight", init_ws)?
                .to_vec2::<f32>()?[0]
        );
        let model = seq()
            .add(linear(space, 64, vb.pp("linear_in"))?)
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
                .forward(&state.detach().unsqueeze(0).unwrap())
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
            let select = weighted_sample(action_probs, rng)? as i64;

            debug!("Action Probability Selected {:?}", select);
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
