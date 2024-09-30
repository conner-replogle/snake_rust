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
use std::fs::File;
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
                GameState::Running => 0.1,
                GameState::AteFood => 1.0,
                GameState::WastedMoves => -0.1,
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
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    let mut varmap = VarMap::new();
    // if (true) {
    //     // varmap.load("snake_model.st")?;
    // }

    let mut model = Model::new(&varmap, &device, SIZE * SIZE, 4)?;

    let mut start_time = Instant::now();

    let out_dir = std::path::Path::new("models");

    let optimizer_params = ParamsAdamW {
        lr: 0.001,
        weight_decay: 0.00,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), optimizer_params)?;

    let (state_tx, state_rx) = std::sync::mpsc::channel::<(Tensor, Sender<Direction>)>();

    let mut rng = ThreadRng::default();
    let epochs = 1000;

    let out_file = File::create("train_log.csv").unwrap();
    let mut wtr = csv::Writer::from_writer(out_file);

    for epoch_idx in 0..epochs {
        info!(
            "Starting EPOCH {epoch_idx}/{epochs} LastEpochTime: {}",
            start_time.elapsed().as_secs_f32()
        );
        start_time = Instant::now();

        let mut steps: Vec<Step> = Vec::new();
        let mut handles = Vec::new();

        for i in 0..8 {
            let device = device.clone();

            let state_tx = state_tx.clone();
            let handle = std::thread::spawn(move || game_thread(state_tx, &device, 500));
            handles.push(handle);
        }
        loop {
            if handles.iter().all(|a| a.is_finished()) {
                debug!("All threads finished");
                break;
            }

            if let Ok((tensor, send)) = state_rx.try_recv() {
                let input = model.predict(&tensor, &mut rng).unwrap();
                send.send(Direction::try_from(input).unwrap()).unwrap();
            }
        }
        for handle in handles {
            let new_steps = handle.join().unwrap().unwrap();
            info!("Thread finished with {} steps", new_steps.len());
            steps.extend(new_steps);
        }

        let mut learn = model.learn(steps, &device, &mut opt)?;
        learn.time = epoch_idx;
        info!("{learn:?}");

        wtr.serialize(learn).unwrap();

        wtr.flush().unwrap();
        if (epoch_idx + 1) % 100 == 0 {
            varmap.save(out_dir.join(format!("snake_model_{}.st",epoch_idx+1)))?;
        }
    }
    varmap.save("snake_model.st")?;

    Ok(())
}
