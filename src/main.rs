mod game;

use macroquad::prelude::*;
use crate::game::Game;

#[macroquad::main("BasicShapes")]
async fn main() {
    let game = Game::new();
    loop {
        clear_background(BLACK);
        game.draw();

        next_frame().await
    }
}