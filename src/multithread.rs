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
use model::LearnOutput;
use serde::de;
use std::fs::{self, File};
use std::sync::mpsc::{self, Sender};
use std::time::Instant;

use crate::game::{Direction, Game, GameState};
use crate::model::{Model, Step};
use macroquad::prelude::*;

use tracing::{debug, info, trace};

pub fn game_thread(
    state: Sender<((Tensor, Tensor), Sender<Direction>)>,
    device: &Device,
    amount: usize,
    width: usize,
    height: usize,
) -> Result<Vec<Step>> {
    let mut game = Game::new(width, height);

    let mut steps: Vec<Step> = Vec::new();
    let mut rng = ThreadRng::default();
    game.reset(&mut rng);
    let mut games = 0;
    loop {
        let tensor = get_model_input_from_game(&game, device).unwrap();
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
            games += 1;

            if steps.len() > amount {
                steps.push(step);
                break;
            }

            game.reset(&mut rng);
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

    let mut model = Model::new(&varmap, &device, SIZE, 4)?;
    if let Some(model) = model_path {
        varmap.load(model)?;
    }
    let mut start_time = Instant::now();

    let optimizer_params = ParamsAdamW {
        // lr:0.0001,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), optimizer_params)?;

    let (state_tx, state_rx) = std::sync::mpsc::channel::<((Tensor, Tensor), Sender<Direction>)>();

    let mut rng = ThreadRng::default();
    let epochs = 10_000;

    let out_file = File::create(out_dir.join("train_log.csv")).unwrap();
    let mut wtr = csv::Writer::from_writer(out_file);
    let mut last_x_deltas = Vec::new();
    for epoch_idx in 0..epochs {
        let delta = start_time.elapsed().as_secs_f32();
        last_x_deltas.push(delta);
        if (last_x_deltas.len() as u32) > 10 {
            last_x_deltas.remove(0);
        }
        let eta = last_x_deltas.iter().sum::<f32>() / last_x_deltas.len() as f32
            * (epochs - epoch_idx) as f32;

        info!(
            "Starting EPOCH {epoch_idx}/{epochs} LastEpochTime: {} Eta: {} mins",
            delta,
            eta / 60.0
        );
        start_time = Instant::now();

        let mut steps = Vec::new();
        let mut handles = Vec::new();
        const SIZES: [(usize, usize); 4] = [(5, 5), (8, 8), (10, 10), (15, 15)];
        for i in 0..4 {
            let device = device.clone();

            let state_tx = state_tx.clone();
            let handle = std::thread::spawn(move || {
                game_thread(state_tx, &device, 200, SIZES[i].0, SIZES[i].1)
            });
            handles.push((i, handle));
        }
        loop {
            let mut i = 0;
            if handles.len() == 0 {
                break;
            }
            while i < handles.len() {
                if !handles[i].1.is_finished() {
                    i += 1;
                    continue;
                }
                let (a, handle) = handles.remove(i);
                let new_steps = handle.join().unwrap().unwrap();
                debug!("Thread {a} finished with {} steps", new_steps.len());
                steps.push((a, new_steps));
            }

            match state_rx.try_recv() {
                Ok((tensor, send)) => {
                    let input = model.predict(&tensor, &mut rng).unwrap();
                    send.send(Direction::try_from(input).unwrap()).unwrap();
                }
                Err(err) => {
                    if err != std::sync::mpsc::TryRecvError::Empty {
                        panic!("Error {:?}", err);
                    }
                }
            }
        }

        let mut total_learn = LearnOutput::default();
        for (a, step) in steps.into_iter() {
            let mut learn = model.learn(step, &device, &mut opt)?;
            learn.time = epoch_idx;

            info!("{:?} {learn:?}", SIZES[a]);
            wtr.serialize(learn).unwrap();
        }

        wtr.flush().unwrap();

        if (epoch_idx + 1) % 100 == 0 {
            info!("Saving model");
            varmap.save(out_dir.join(format!("snake_model_{}.st", epoch_idx + 1)))?;
        }
    }
    varmap.save(out_dir.join(format!("snake_model.st")))?;

    Ok(())
}
