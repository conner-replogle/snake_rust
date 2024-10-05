use ::rand::rngs::ThreadRng;
use candle_core::Device;
use candle_nn::{VarBuilder, VarMap};

use chrono::{Duration, TimeDelta};
use macroquad::prelude::*;
use snake_rust::connector::get_model_input_from_game;
use snake_rust::game::{Direction, Game, GameState};
use snake_rust::model::{Model, Step};
use snake_rust::timer::Timer;
use std::env::var;
use std::ops::Index;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tracing::{debug, info};

const SIZE: usize = 25;

static MODEL_FILE: &[u8; include_bytes!("../snake_model.st").len()] =
    include_bytes!("../snake_model.st");

#[macroquad::main("Snake")]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = Model::load(&device, MODEL_FILE)?;

    let mut timer = Timer::new(TimeDelta::milliseconds(100));
    let mut rng = ThreadRng::default();
    loop {
        let mut game = Game::new(SIZE, SIZE);
        game.reset(&mut rng);
        loop {
            if timer.tick() {
                let tensor = get_model_input_from_game(&game, &device)?;

                let input = model.predict(&tensor, &mut rng).unwrap();
                game.send_input(Direction::try_from(input).unwrap());
                let out = game.step(&mut rng);

                if out != GameState::Running && out != GameState::AteFood && out != GameState::WastedMoves {
                    trace!("GameState {:?} score: {:?}", out, game.score);

                    game.reset(&mut rng);
                }
            }

            game.draw();
            next_frame().await;
        }
    }
    Ok(())
}
