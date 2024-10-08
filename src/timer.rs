use chrono::{DateTime, Duration, Local};

pub struct Timer {
    last_time: DateTime<Local>,
    duration: Duration,
}
impl Timer {
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            last_time: Local::now(),
        }
    }
    pub fn set_duration(&mut self, duration: Duration) {
        self.duration = duration;
    }
    pub fn tick(&mut self) -> bool {
        let time = Local::now();
        if time - self.last_time > self.duration {
            self.last_time = time;
            return true;
        }

        return false;
    }
}
