use ::rand::rngs::ThreadRng;
use ::rand::Rng;
use macroquad::color::{GRAY, GREEN, RED};
use macroquad::prelude::DARKGREEN;
use macroquad::rand;
use macroquad::shapes::draw_rectangle;
use macroquad::window::{screen_height, screen_width};
use nalgebra::{DMatrix, SMatrix};
use num_traits::{ToPrimitive, Zero};
use std::ops::IndexMut;
use tracing::{debug, trace};

type State = DMatrix<u8>;

pub struct Snake {
    head: (isize, isize),        //index
    bodies: Vec<(isize, isize)>, //indexes
    direction: Direction,
}
#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone)]
enum CellState {
    Blank = 0,
    SnakeHead = 1,
    SnakeBody = 2,
    Food = 3,
}
impl From<u8> for CellState {
    fn from(value: u8) -> Self {
        match value {
            0 => CellState::Blank,
            1 => CellState::SnakeHead,
            2 => CellState::SnakeBody,
            3 => CellState::Food,
            _ => panic!("Invalid value"),
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum GameState {
    Running,
    DiedByWall,
    DiedBySelf,
    WastedMoves,
    AteFood,
    Won,
}
impl GameState {
    pub fn reward(&self) -> f32 {
        return match self {
            GameState::Running => 0.0,
            GameState::AteFood => 2.0,
            GameState::WastedMoves => -0.1,
            GameState::Won => 5.0,
            _ => -1.0,
        };
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl TryFrom<usize> for Direction {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        return Ok(match value {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => {
                panic!("BAD");
            }
        });
    }
}

pub struct Game {
    state: State,
    pub size: (usize, usize),
    snake: Snake,
    food: (usize, usize),
    pub score: usize,
    pub moves_since_last_meal: usize,
}
impl Game {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            state: DMatrix::from_element(w, h, CellState::Blank as u8),
            snake: Snake {
                head: ((w / 2) as isize, (h / 2) as isize),
                bodies: vec![((w / 2) as isize, (h / 2) as isize - 1)],
                direction: Direction::Right,
            },
            food: (0, 0),
            score: 0,
            moves_since_last_meal: 0,
            size: (w, h),
        }
    }

    pub fn reset(&mut self, rng: &mut ThreadRng) {
        self.score = 0;
        self.snake.head = (
            rand::gen_range(1, self.size.0 as isize - 1),
            rand::gen_range(1, self.size.1 as isize - 1),
        );
        self.moves_since_last_meal = 0;
        self.snake.bodies = vec![(self.snake.head.0 - 1, self.snake.head.1)];
        self.spawn_food(rng);
        self.snake.direction = Direction::Right;
        self.generate_state();
    }
}
impl Game {
    pub fn get_state(&self) -> DMatrix<u8> {
        return self.state.clone();
    }

    pub fn get_snake_state(&self) -> [f32; 12] {
        let state = self.get_state();
        let (head_x, head_y) = self.snake.head;
        let (head_x, head_y) = (head_x as usize, head_y as usize);
        let mut snake_state = [0.0; 12];
        let mut obstacle_state = [0; 4];

        trace!("Head: {:?} ", (head_x, head_y));

        // Check up
        if head_y == 0 || state[(head_x, head_y - 1)] == CellState::SnakeBody as u8 {
            obstacle_state[0] = 1;
        }
        // Check down
        if head_y == self.size.1 - 1 || state[(head_x, head_y + 1)] == CellState::SnakeBody as u8 {
            obstacle_state[1] = 1;
        }
        // Check left
        if head_x == 0 || state[(head_x - 1, head_y)] == CellState::SnakeBody as u8 {
            obstacle_state[2] = 1;
        }
        // Check right
        if head_x == self.size.0 - 1 || state[(head_x + 1, head_y)] == CellState::SnakeBody as u8 {
            obstacle_state[3] = 1;
        }
        trace!("Obstacle Dir: {:?}", obstacle_state);
        let mut food_state = [0; 4];
        let (food_x, food_y) = self.food;

        // Check up
        if food_y < head_y {
            food_state[0] = 1;
        }
        // Check down
        if food_y > head_y {
            food_state[1] = 1;
        }
        // Check left
        if food_x < head_x {
            food_state[2] = 1;
        }
        // Check right
        if food_x > head_x {
            food_state[3] = 1;
        }

        trace!("Food Dir: {:?}", food_state);

        let mut food_distance = [0.0; 4];
        if food_state[0] == 1 {
            food_distance[0] = (head_y - food_y) as f32 / self.size.1 as f32;
        }
        if food_state[1] == 1 {
            food_distance[1] = (food_y - head_y) as f32 / self.size.1 as f32;
        }
        if food_state[2] == 1 {
            food_distance[2] = (head_x - food_x) as f32 / self.size.0 as f32;
        }
        if food_state[3] == 1 {
            food_distance[3] = (food_x - head_x) as f32 / self.size.0 as f32;
        }
        trace!("Food Distance: {:?}", food_distance);

        snake_state[8..12].copy_from_slice(&food_distance);
        snake_state[4..8].copy_from_slice(&food_state.map(|a| a.to_f32().unwrap()));
        snake_state[0..4].copy_from_slice(&obstacle_state.map(|a| a.to_f32().unwrap()));

        snake_state
    }
    pub fn send_input(&mut self, direction: Direction) {
        self.snake.direction = direction
    }
    pub fn spawn_food(&mut self, rng: &mut ThreadRng) -> GameState {
        self.generate_state();

        if (self.snake.bodies.len() + 1) >= (self.size.0 * self.size.1) {
            return GameState::Won;
        }
        self.food = (rng.gen_range(0..self.size.0), rng.gen_range(0..self.size.1));
        while *self.state.index(self.food) != CellState::Blank as u8 {
            self.food = (rng.gen_range(0..self.size.0), rng.gen_range(0..self.size.1));
        }
        self.moves_since_last_meal = 0;
        return GameState::AteFood;
    }
    pub fn step(&mut self, rng: &mut ThreadRng) -> GameState {
        let old_head = self.snake.head;

        let mut head = self.snake.head;
        let (x, y) = match self.snake.direction {
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
            Direction::Up => (0, -1),
        };
        head.0 += x;
        head.1 += y;

        self.moves_since_last_meal += 1;
        if head.0 >= self.size.0 as isize
            || head.1 >= self.size.1 as isize
            || head.0 < 0
            || head.1 < 0
        {
            return GameState::DiedByWall;
        }
        if self.snake.bodies.contains(&head) {
            return GameState::DiedBySelf;
        }

        if (self.moves_since_last_meal as f32)
            >= ((self.size.0 * self.size.1) as f32 * 0.8).max(25.0)
        {
            return GameState::WastedMoves;
        }

        self.snake.bodies.push(old_head);
        self.snake.head = head;

        if self.snake.head.0 as usize == self.food.0 && self.snake.head.1 as usize == self.food.1 {
            let out = self.spawn_food(rng);
            self.generate_state();
            self.score += 1;

            return out;
        } else {
            self.snake.bodies.remove(0);
        }
        self.generate_state();

        return GameState::Running;
    }
    fn generate_state(&mut self) {
        self.state = DMatrix::from_element(self.size.0, self.size.1, CellState::Blank as u8);

        *self.state.index_mut(self.food) = CellState::Food as u8;
        *self
            .state
            .index_mut((self.snake.head.0 as usize, self.snake.head.1 as usize)) =
            CellState::SnakeHead as u8;
        for body in self
            .snake
            .bodies
            .iter()
            .map(|a| (a.0 as usize, a.1 as usize))
        {
            *self.state.index_mut(body) = CellState::SnakeBody as u8;
        }
    }
}

impl Game {
    pub fn draw(&self) {
        let width = screen_width();
        let height = screen_height();
        let padding = 10.0;
        let gap: f32 = 1.10; //% of cell_size
        let cell_size = (width.min(height) - 2.0 * padding) * (2.0 - gap)
            / (self.size.0.min(self.size.1) as f32);

        for i in 0..(self.size.0 * self.size.1) {
            let (x, y) = (i % self.size.0, i / self.size.1);
            let (pix_x, pix_y) = (
                padding + (cell_size * gap) * x as f32,
                padding + (cell_size * gap) * y as f32,
            );
            let item = &self.state[i];

            let color = match CellState::try_from(*item).unwrap() {
                CellState::Blank => GRAY,
                CellState::Food => GREEN,
                CellState::SnakeHead => RED,
                CellState::SnakeBody => DARKGREEN,
            };
            draw_rectangle(pix_x, pix_y, cell_size, cell_size, color)
        }
    }
}
