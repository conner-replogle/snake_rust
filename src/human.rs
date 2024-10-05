use ::rand::rngs::ThreadRng;
use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, MetalDevice, Result, Tensor, Var};
use candle_nn::{init, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use snake_rust::config::SIZE;
use snake_rust::connector::get_model_input_from_game;
use std::env::var;
use std::ops::Index;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use macroquad::prelude::*;
use snake_rust::game::{Direction, Game, GameState};
use snake_rust::model::{Model, Step};
use snake_rust::timer::Timer;

use tracing::{debug, info};

#[macroquad::main("Snake")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;

    let mut rng = ThreadRng::default();
    for _ in 0..100 {
        let mut game = Game::new(SIZE, SIZE);
        let mut steps: Vec<Step> = Vec::new();
        game.reset(&mut rng);
        loop {
            if is_key_down(KeyCode::A) {
                game.send_input(Direction::Left);
            }
            if is_key_down(KeyCode::D) {
                game.send_input(Direction::Right);
            }
            if is_key_down(KeyCode::W) {
                game.send_input(Direction::Up);
            }
            if is_key_down(KeyCode::S) {
                game.send_input(Direction::Down);
            }

            if is_key_released(KeyCode::G) {
                let tensor = get_model_input_from_game(&game, &device)?;

                let out = game.step(&mut rng);

                let step = Step {
                    input: tensor,
                    terminated: out != GameState::Running && out != GameState::AteFood,
                    action: 0 as i64,
                    reward: out.reward(),
                    state: out,
                };
                trace!("Step: {:?}", step);

                if step.terminated {
                    trace!("GameState {:?} score: {:?}", out, game.score);

                    game.reset(&mut rng);
                    if (steps.len()) > 40 {
                        steps.push(step);
                        break;
                    }
                }
                let tensor = get_model_input_from_game(&game, &device)?;
                steps.push(step);
            }

            game.draw();
            next_frame().await;
        }
    }
    Ok(())
}
