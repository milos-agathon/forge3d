use std::sync::{mpsc, Arc, Mutex};
use std::thread;

pub(crate) type Job = Box<dyn FnOnce() + Send + 'static>;

pub(crate) struct ThreadPool {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Job>,
}

struct Worker {
    #[allow(dead_code)]
    id: usize,
    handle: Option<thread::JoinHandle<()>>,
}

impl ThreadPool {
    pub(crate) fn new(size: usize) -> ThreadPool {
        assert!(size > 0);

        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));

        let mut workers = Vec::with_capacity(size);

        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver)));
        }

        ThreadPool { workers, sender }
    }

    pub(crate) fn execute<F>(&self, f: F) -> Result<(), mpsc::SendError<Job>>
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.send(job)
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Close the channel
        drop(self.sender.clone());

        // Wait for all workers to finish
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                handle
                    .join()
                    .unwrap_or_else(|_| eprintln!("Worker thread panicked"));
            }
        }
    }
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<mpsc::Receiver<Job>>>) -> Worker {
        let handle = thread::spawn(move || loop {
            let receiver = receiver.lock().unwrap();
            match receiver.recv() {
                Ok(job) => {
                    drop(receiver); // Release lock before executing
                    job();
                }
                Err(_) => break, // Channel closed
            }
        });

        Worker {
            id,
            handle: Some(handle),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new(4);
        // Pool should be created successfully
        drop(pool); // Test cleanup
    }
}
