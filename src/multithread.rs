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
use std::fs::{self, File};
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
    let mut rng = ThreadRng::default();
    game.reset(&mut rng);
    loop {
        let tensor = get_model_input_from_game(&game, device)?;
        let (send, recv) = mpsc::channel();
        state.send((tensor.clone(), send)).unwrap();
        let input = recv.recv().unwrap();

        game.send_input(input);
        let out = game.step(&mut rng);

        let step = Step {
            input: tensor,
            terminated: out != GameState::Running && out != GameState::AteFood,
            action: input as i64,
            reward: out.reward(),
            state: out,
        };
        trace!("Step: {:?}", step);

        if step.terminated {
            trace!("GameState {:?} score: {:?}", out, game.score);
            game.reset(&mut rng);

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

    let path = std::env::args().nth(1).expect("no name given");

    let out_dir = std::path::Path::new("models");
    let out_dir = out_dir.join(path);
    fs::create_dir(&out_dir);

    let model_path = std::env::args().nth(2);

    let mut model = Model::new(&varmap, &device, SIZE , 4)?;
    if let Some(model) = model_path {
        varmap.load(model)?;
    }
    let mut start_time = Instant::now();

    let optimizer_params = ParamsAdamW {
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), optimizer_params)?;

    let (state_tx, state_rx) = std::sync::mpsc::channel::<(Tensor, Sender<Direction>)>();

    let mut rng = ThreadRng::default();
    let epochs = 10_000;

    let out_file = File::create(out_dir.join("train_log.csv")).unwrap();
    let mut wtr = csv::Writer::from_writer(out_file);

    for epoch_idx in 0..epochs {
        info!(
            "Starting EPOCH {epoch_idx}/{epochs} LastEpochTime: {}",
            start_time.elapsed().as_secs_f32()
        );
        start_time = Instant::now();

        let mut steps: Vec<Step> = Vec::new();
        let mut handles = Vec::new();

        for _ in 0..8 {
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
            info!("Saving model");
            varmap.save(out_dir.join(format!("snake_model_{}.st", epoch_idx + 1)))?;
        }
    }
    varmap.save(out_dir.join(format!("snake_model.st")))?;

    Ok(())
}
