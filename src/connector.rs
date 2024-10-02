use crate::game::Game;
use candle_core::{Device, Result, Tensor};
use tracing::debug;

pub fn get_model_input_from_game(game: &Game, device: &Device) -> Result<(Tensor, Tensor)> {
    let state: Vec<f32> = game
        .get_state()
        .into_iter()
        .map(|a| {
            let mut state: [f32; 3] = [0.0, 0.0, 0.0];
            if *a == 0 {
                return state;
            }
            state[*a as usize - 1] = 1.0;
            return state;
        })
        .flatten()
        .collect();
    let tensor = Tensor::from_vec(state, (game.size.0, game.size.1, 3), &device)?;
    // debug!("Shape: {:?} Before {:?}", tensor.shape(),tensor.to_vec3::<f32>()?);
    let conv_tensor = tensor.transpose(0, 2)?.transpose(1, 2)?;
    let snake_state = game.get_snake_state();

    // debug!("Shape: {:?} After {:?}", output_tensor.shape(),output_tensor.to_vec3::<f32>()?);

    return Ok((conv_tensor, Tensor::from_slice(&snake_state, 12, device)?));
}
