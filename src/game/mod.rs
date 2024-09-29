use macroquad::color::{GRAY, GREEN, RED};
use macroquad::prelude::DARKGREEN;
use macroquad::rand;
use macroquad::shapes::draw_rectangle;
use macroquad::window::{screen_height, screen_width};
use nalgebra::SMatrix;
use num_traits::Zero;
use std::ops::IndexMut;

type State<const W: usize, const L: usize> = SMatrix<u8, W, L>;

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

#[derive(PartialEq, Debug)]
pub enum GameState {
    Running,
    DiedByWall,
    DiedBySelf,
    AteFood,
}

#[derive(Debug)]
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

pub struct Game<const W: usize, const L: usize> {
    state: State<W, L>,
    snake: Snake,
    food: (usize, usize),
    pub score: usize,
}
impl<const W: usize, const L: usize> Game<W, L> {
    pub fn new() -> Self {
        let mut matrix: SMatrix<u8, W, L> = SMatrix::zeros();


        Self {
            state: matrix,
            snake: Snake {
                head: ((W / 2) as isize, (L / 2) as isize),
                bodies: vec![((W / 2) as isize, (L / 2) as isize -1)],
                direction: Direction::Right,
            },
            food: (0, 0),
            score: 0,
        }
    }

    pub fn reset(&mut self) {
        self.score = 0;
        self.snake.head = (
            rand::gen_range(1, W as isize - 1),
            rand::gen_range(1, L as isize - 1),
        );
        self.snake.bodies = vec![(self.snake.head.0 - 1, self.snake.head.1)];
        self.spawn_food();
        self.snake.direction = Direction::Right;
        self.generate_state();
    }
}
impl<const W: usize, const L: usize> Game<W, L> {
    pub fn get_state(&self) -> SMatrix<u8, W, L> {
        return self.state;
    }
    pub fn send_input(&mut self, direction: Direction) {
        self.snake.direction = direction
    }
    pub fn spawn_food(&mut self) {
        self.food = (rand::gen_range(0, W), rand::gen_range(0, L));
    }
    pub fn step(&mut self) -> GameState {
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

        if head.0 >= W as isize || head.1 >= L as isize || head.0 < 0 || head.1 < 0 {
            return GameState::DiedByWall;
        }
        if self.snake.bodies.contains(&head) {
            return GameState::DiedBySelf;
        }

        self.snake.bodies.push(old_head);
        self.snake.head = head;

        if self.snake.head.0 as usize == self.food.0 && self.snake.head.1 as usize == self.food.1 {
            self.spawn_food();
            self.generate_state();
            self.score += 1;
            return GameState::AteFood;
        } else {
            self.snake.bodies.remove(0);
        }
        self.generate_state();

        return GameState::Running;
    }
    fn generate_state(&mut self) {
        self.state.set_zero();

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

impl<const W: usize, const L: usize> Game<W, L> {
    pub fn draw(&self) {
        let width = screen_width();
        let height = screen_height();
        let padding = 10.0;
        let gap: f32 = 1.10; //% of cell_size
        let cell_size = (width.min(height) - 2.0 * padding) * (2.0 - gap) / (W.min(L) as f32);

        for i in 0..(W * L) {
            let (x, y) = (i % W, i / L);
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
