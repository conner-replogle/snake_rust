mod game;
mod model;
mod timer;

use ::rand::rngs::ThreadRng;
use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, MetalDevice, Result, Tensor};
use candle_nn::{init, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::env::var;
use std::ops::Index;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::game::{Direction, Game, GameState};
use crate::model::{Model, Step};
use crate::timer::Timer;
use macroquad::prelude::*;

use tracing::{debug, info};

pub fn game_thread(model: Model, device: &Device) -> Result<Vec<Step>> {
    let mut game = Game::<10, 10>::new();
    let mut steps: Vec<Step> = Vec::new();
    let mut rng = ThreadRng::default();

    game.reset();
    loop {
        let state: Vec<f32> = game
            .get_state()
            .into_iter()
            .map(|a| {
                let mut state: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
                state[*a as usize] = 1.0;
                return state;
            })
            .flatten()
            .collect();
        let tensor = Tensor::from_vec(state, (10, 10, 4), &device)?
            .flatten_all()?
            .unsqueeze(0)?;
        let input = model.predict(&tensor, &mut rng).unwrap();

        game.send_input(Direction::try_from(input).unwrap());
        let out = game.step();

        let step = Step {
            input: tensor,
            terminated: out != GameState::Running && out != GameState::AteFood,
            action: input as i64,
            reward: match out {
                GameState::Running => 5.0,
                GameState::AteFood => 10.0,
                _ => -1.0,
            },
        };
        debug!("Step: {:?}", step);

        if step.terminated {
            debug!("GameState {:?} score: {:?}", out, game.score);
            game.reset();

            if steps.len() > 5000 {
                steps.push(step);
                break;
            }
        }

        steps.push(step);
    }
    return Ok(steps);
}

#[macroquad::main("Snake")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let device = Device::new_cuda(0)?;
    let varmap = VarMap::new();

    let mut model = Model::new(&varmap, &device, 10 * 10 * 4, 4)?;
    let mut timer = Timer::new(Duration::from_millis(0));
    let mut draw = true;
    let mut start_time = Instant::now();
    let optimizer_params = ParamsAdamW {
        lr: 0.01,
        weight_decay: 0.01,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), optimizer_params)?;

    for epoch_idx in 0..100 {
        info!(
            "Starting EPOCH {epoch_idx} SecondsPerEpoch{}",
            start_time.elapsed().as_secs_f32()
        );

        let mut game = Game::<10, 10>::new();
        let mut rng = ThreadRng::default();
        let mut steps = Vec::new();

        game.reset();
        loop {
            if timer.tick() {
                let state: Vec<f32> = game
                    .get_state()
                    .into_iter()
                    .map(|a| {
                        let mut state: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
                        state[*a as usize] = 1.0;
                        return state;
                    })
                    .flatten()
                    .collect();
                let tensor = Tensor::from_vec(state, (10, 10, 4), &device)?
                    .flatten_all()?
                    .unsqueeze(0)?;
                let input = model.predict(&tensor, &mut rng).unwrap();

                game.send_input(Direction::try_from(input).unwrap());
                let out = game.step();

                let step = Step {
                    input: tensor,
                    terminated: out != GameState::Running && out != GameState::AteFood,
                    action: input as i64,
                    reward: match out {
                        GameState::Running => 5.0,
                        GameState::AteFood => 10.0,
                        _ => -1.0,
                    },
                };
                debug!("Step: {:?}", step);
                steps.push(step.clone());

                if step.terminated {
                    debug!("GameState {:?} score: {:?}", out, game.score);
                    game.reset();
                    if steps.len() > 300 {
                        break;
                    }
                }
            }

            game.draw();
            next_frame().await;
        }
        model.learn(steps, &device, &mut opt)?;
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let init_ws = init::DEFAULT_KAIMING_NORMAL;
        tracing::info!(
            "Creating Model {:?}",
            vb.pp("linear_in")
                .get_with_hints((64, 400), "weight", init_ws)?
                .to_vec2::<f32>()?[0]
        );
    }
    varmap.save("snake_model.st")?;
    Ok(())
}
