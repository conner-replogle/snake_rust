mod config;
mod connector;
mod game;
mod model;
mod timer;
use ::rand::rngs::ThreadRng;
use candle_core::{DType, Device, MetalDevice, Result, Tensor};
use candle_nn::{init, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use config::SIZE;
use connector::get_model_input_from_game;
use std::sync::mpsc::{self, Sender};
use std::time::Instant;

use crate::game::{Direction, Game, GameState};
use crate::model::{Model, Step};
use macroquad::prelude::*;

use tracing::{debug, info, trace};

pub fn game_thread(
    state: Sender<(Tensor, Sender<Direction>)>,
    device: &Device,
    amount: usize,
) -> Result<Vec<Step>> {
    let mut game = Game::<SIZE, SIZE>::new();
    let mut steps: Vec<Step> = Vec::new();

    game.reset();
    loop {
        let tensor = get_model_input_from_game(&game, device)?;
        let (send, recv) = mpsc::channel();
        state.send((tensor.clone(), send)).unwrap();
        let input = recv.recv().unwrap();

        game.send_input(input);
        let out = game.step();

        let step = Step {
            input: tensor,
            terminated: out != GameState::Running && out != GameState::AteFood,
            action: input as i64,
            reward: match out {
                GameState::Running => 0.01,
                GameState::AteFood => 5.0,
                _ => -1.0,
            },
        };
        trace!("Step: {:?}", step);

        if step.terminated {
            trace!("GameState {:?} score: {:?}", out, game.score);
            game.reset();

            if steps.len() > amount {
                steps.push(step);
                break;
            }
        }

        steps.push(step);
    }
    return Ok(steps);
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let device = Device::new_metal(0)?;
    let varmap = VarMap::new();

    let mut model = Model::new(&varmap, &device, SIZE * SIZE, 4)?;

    let mut start_time = Instant::now();
    let optimizer_params = ParamsAdamW {
        lr: 0.01,
        weight_decay: 0.01,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), optimizer_params)?;

    let (state_tx, state_rx) = std::sync::mpsc::channel::<(Tensor, Sender<Direction>)>();

    let mut rng = ThreadRng::default();
    for epoch_idx in 0..100 {
        info!(
            "Starting EPOCH {epoch_idx} SecondsPerEpoch{}",
            start_time.elapsed().as_secs_f32()
        );
        start_time = Instant::now();

        let mut steps: Vec<Step> = Vec::new();
        let mut handles = Vec::new();

        for i in 0..8 {
            debug!("Starting thread {i}");
            let device = device.clone();

            let state_tx = state_tx.clone();
            let handle = std::thread::spawn(move || game_thread(state_tx, &device, 5000));
            handles.push(handle);
        }
        loop {
            if handles.iter().all(|a| a.is_finished()) {
                debug!("All threads finished");
                break;
            }

            if let Ok((tensor, send)) = state_rx.try_recv() {
                trace!("Received state");
                let input = model.predict(&tensor, &mut rng).unwrap();
                send.send(Direction::try_from(input).unwrap()).unwrap();
            }
        }
        for handle in handles {
            steps.extend(handle.join().unwrap().unwrap());
        }

        model.learn(steps, &device, &mut opt)?;
    }
    varmap.save("snake_model.st")?;
    Ok(())
}
