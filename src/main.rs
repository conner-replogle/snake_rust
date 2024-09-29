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
    let device = Device::new_metal(0)?;
    let mut varmap = VarMap::new();
    varmap.load("snake_model.st")?;

    let mut model = Model::new(&varmap, &device, 10 * 10 * 4, 4)?;
    let mut timer = Timer::new(Duration::from_millis(300));
    let mut draw = true;
    let mut start_time = Instant::now();

    for epoch_idx in 0..100 {
        let mut game = Game::<10, 10>::new();
        let mut rng = ThreadRng::default();

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

                if step.terminated {
                    debug!("GameState {:?} score: {:?}", out, game.score);
                    game.reset();
                }
            }

            game.draw();
            next_frame().await;
        }
    }
    Ok(())
}
