use ::rand::rngs::ThreadRng;
use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, MetalDevice, Result, Tensor, Var};
use candle_nn::{init, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use chrono::TimeDelta;
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
    let path = std::env::args().nth(1).expect("no name given");
    let mut varmap = VarMap::new();

    let model = Model::new(&varmap, &device)?;

    varmap.load(path)?;
    let mut timer = Timer::new(TimeDelta::milliseconds(100));

    let mut rng = ThreadRng::default();
    for _ in 0..100 {
        let mut game = Game::new(SIZE, SIZE);
        let mut steps: Vec<Step> = Vec::new();
        game.reset(&mut rng);
        loop {
            if timer.tick() {
                let tensor = get_model_input_from_game(&game, &device)?;

                let input = model.predict(&tensor, &mut rng).unwrap();
                game.send_input(Direction::try_from(input).unwrap());
                let out = game.step(&mut rng);

                let step = Step {
                    input: tensor,
                    terminated: out != GameState::Running && out != GameState::AteFood,
                    action: input as i64,
                    reward: out.reward(),
                    state: out,
                };
                trace!("Step: {:?}", step);

                if step.terminated || is_key_down(KeyCode::Space) {
                    trace!("GameState {:?} score: {:?}", out, game.score);

                    game.reset(&mut rng);
                    if (steps.len()) > 40 {
                        steps.push(step);
                        break;
                    }
                }
                steps.push(step);
            }

            game.draw();
            next_frame().await;
        }
    }
    Ok(())
}
