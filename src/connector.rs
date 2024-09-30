use candle_core::{Device, Result, Tensor};

use crate::game::Game;

pub fn get_model_input_from_game<const W: usize, const L: usize>(
    game: &Game<W, L>,
    device: &Device,
) -> Result<Tensor> {
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
    let tensor = Tensor::from_vec(state, (4, W, L), &device)?;
    return Ok(tensor);
}
