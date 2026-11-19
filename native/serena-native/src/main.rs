use std::fs;
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Parser)]
#[command(name = "serena-native", version, about)]
struct Cli {
    #[command(subcommand)]
    command: NativeCommand,
}

#[derive(Subcommand)]
enum NativeCommand {
    /// Run the persistent Serena daemon in the foreground.
    Serve(ServeArgs),
    /// Send one newline-delimited JSON request to a daemon.
    Request(RequestArgs),
}

#[derive(Args)]
struct ServeArgs {
    #[arg(long)]
    project: PathBuf,
    #[arg(long)]
    socket: PathBuf,
    #[arg(long)]
    pid_file: PathBuf,
    #[arg(long)]
    metadata_file: PathBuf,
    #[arg(long)]
    python: PathBuf,
    #[arg(long, default_value = "serena.daemon_worker")]
    worker_module: String,
    #[arg(long)]
    worker_config_json: String,
}

#[derive(Args)]
struct RequestArgs {
    #[arg(long)]
    socket: PathBuf,
    #[arg(long)]
    request_json: String,
    #[arg(long, default_value_t = 300_000)]
    timeout_ms: u64,
}

#[derive(Serialize)]
struct DaemonMetadata<'a> {
    version: u32,
    pid: u32,
    worker_pid: Option<u32>,
    project: &'a Path,
    socket: &'a Path,
    started_at_unix_ms: u64,
}

struct StateFileGuard {
    socket: PathBuf,
    pid_file: PathBuf,
    metadata_file: PathBuf,
}

impl StateFileGuard {
    fn new(args: &ServeArgs) -> Self {
        Self {
            socket: args.socket.clone(),
            pid_file: args.pid_file.clone(),
            metadata_file: args.metadata_file.clone(),
        }
    }
}

impl Drop for StateFileGuard {
    fn drop(&mut self) {
        remove_if_exists(&self.socket);
        remove_if_exists(&self.pid_file);
        remove_if_exists(&self.metadata_file);
    }
}

struct WorkerProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    shutdown_sent: bool,
}

impl WorkerProcess {
    fn spawn(args: &ServeArgs) -> Result<Self> {
        let mut child = Command::new(&args.python)
            .arg("-m")
            .arg(&args.worker_module)
            .arg("--config-json")
            .arg(&args.worker_config_json)
            .env("PYTHONUNBUFFERED", "1")
            .env("SERENA_USAGE_REPORTING", "false")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start Python worker with {}",
                    args.python.display()
                )
            })?;

        let stdin = child
            .stdin
            .take()
            .context("Python worker stdin was not piped")?;
        let stdout = child
            .stdout
            .take()
            .context("Python worker stdout was not piped")?;
        let mut worker = Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            shutdown_sent: false,
        };
        if let Err(error) = worker.wait_until_ready() {
            worker.shutdown();
            return Err(error);
        }
        Ok(worker)
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn wait_until_ready(&mut self) -> Result<()> {
        let mut line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut line)
            .context("failed to read Python worker readiness message")?;
        if bytes == 0 {
            let status = self
                .child
                .try_wait()
                .context("failed to inspect Python worker status")?;
            bail!("Python worker exited before becoming ready: {status:?}");
        }

        let message: Value = serde_json::from_str(line.trim_end()).with_context(|| {
            format!(
                "invalid Python worker readiness message: {}",
                line.trim_end()
            )
        })?;
        if message.get("event").and_then(Value::as_str) == Some("ready") {
            return Ok(());
        }
        if message.get("event").and_then(Value::as_str) == Some("fatal") {
            let error = message
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown worker initialization error");
            bail!("Python worker failed to initialize: {error}");
        }
        bail!("unexpected Python worker readiness message: {message}")
    }

    fn request(&mut self, request: &str) -> Result<String> {
        self.stdin
            .write_all(request.trim_end().as_bytes())
            .context("failed to write request to Python worker")?;
        self.stdin
            .write_all(b"\n")
            .context("failed to terminate Python worker request")?;
        self.stdin
            .flush()
            .context("failed to flush Python worker request")?;

        let mut response = String::new();
        let bytes = self
            .stdout
            .read_line(&mut response)
            .context("failed to read response from Python worker")?;
        if bytes == 0 {
            bail!("Python worker closed its protocol stream")
        }
        Ok(response)
    }

    fn shutdown(&mut self) {
        if !self.shutdown_sent && self.child.try_wait().ok().flatten().is_none() {
            self.shutdown_sent = true;
            let request = json!({"id": null, "method": "shutdown", "params": {}}).to_string();
            let _ = self.request(&request);
        }

        let deadline = std::time::Instant::now() + Duration::from_secs(8);
        while std::time::Instant::now() < deadline {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => thread::sleep(Duration::from_millis(50)),
                Err(_) => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        NativeCommand::Serve(args) => serve(args),
        NativeCommand::Request(args) => request(args),
    }
}

fn serve(args: ServeArgs) -> Result<()> {
    ensure_parent_directory(&args.socket)?;
    ensure_parent_directory(&args.pid_file)?;
    ensure_parent_directory(&args.metadata_file)?;
    remove_if_exists(&args.socket);

    let _state_guard = StateFileGuard::new(&args);
    let listener = UnixListener::bind(&args.socket)
        .with_context(|| format!("failed to bind Unix socket {}", args.socket.display()))?;
    listener
        .set_nonblocking(true)
        .context("failed to make daemon socket non-blocking")?;

    let started_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis()
        .try_into()
        .context("current time does not fit in a 64-bit millisecond timestamp")?;
    write_state_files(&args, None, started_at_unix_ms)?;

    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&shutdown_requested);
    ctrlc::set_handler(move || signal_flag.store(true, Ordering::SeqCst))
        .context("failed to install daemon signal handler")?;

    let mut worker = WorkerProcess::spawn(&args)?;
    write_state_files(&args, Some(worker.pid()), started_at_unix_ms)?;

    while !shutdown_requested.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let should_stop = handle_connection(
                    stream,
                    &mut worker,
                    &args.project,
                    &args.socket,
                    started_at_unix_ms,
                )
                .unwrap_or_else(|error| {
                    // A client can time out while the serialized worker is still busy.
                    // A broken client connection must not take down the daemon itself.
                    eprintln!("daemon request failed: {error:#}");
                    false
                });
                if should_stop {
                    shutdown_requested.store(true, Ordering::SeqCst);
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(25))
            }
            Err(error) => return Err(error).context("failed to accept daemon connection"),
        }
    }

    worker.shutdown();
    Ok(())
}

fn handle_connection(
    mut stream: UnixStream,
    worker: &mut WorkerProcess,
    project: &Path,
    socket: &Path,
    started_at_unix_ms: u64,
) -> Result<bool> {
    let mut request_line = String::new();
    BufReader::new(
        stream
            .try_clone()
            .context("failed to clone daemon stream")?,
    )
    .read_line(&mut request_line)
    .context("failed to read daemon request")?;

    let request_value: Value = match serde_json::from_str(request_line.trim_end()) {
        Ok(value) => value,
        Err(error) => {
            let response = error_response(Value::Null, "invalid_request", error.to_string());
            write_response(&mut stream, &response.to_string())?;
            return Ok(false);
        }
    };
    let request_id = request_value.get("id").cloned().unwrap_or(Value::Null);
    let method = match request_value.get("method").and_then(Value::as_str) {
        Some(method) => method,
        None => {
            let response = error_response(
                request_id,
                "invalid_request",
                "request method must be a string",
            );
            write_response(&mut stream, &response.to_string())?;
            return Ok(false);
        }
    };

    let worker_response = match worker.request(&request_line) {
        Ok(response) => response,
        Err(error) => {
            let response = error_response(request_id, "worker_unavailable", error.to_string());
            write_response(&mut stream, &response.to_string())?;
            return Ok(true);
        }
    };
    let response = enrich_response(
        &worker_response,
        method,
        project,
        socket,
        worker.pid(),
        started_at_unix_ms,
    )?;
    if let Err(error) = write_response(&mut stream, &response)
        && !matches!(error.root_cause().downcast_ref::<std::io::Error>(), Some(io_error) if matches!(io_error.kind(), ErrorKind::BrokenPipe | ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted))
    {
        return Err(error);
    }

    if method == "shutdown" {
        worker.shutdown_sent = true;
        return Ok(true);
    }
    Ok(false)
}

fn request(args: RequestArgs) -> Result<()> {
    serde_json::from_str::<Value>(&args.request_json).context("request JSON is invalid")?;
    let mut stream = UnixStream::connect(&args.socket).with_context(|| {
        format!(
            "failed to connect to daemon socket {}",
            args.socket.display()
        )
    })?;
    let timeout = Duration::from_millis(args.timeout_ms);
    stream
        .set_read_timeout(Some(timeout))
        .context("failed to set daemon response timeout")?;
    stream
        .set_write_timeout(Some(timeout))
        .context("failed to set daemon request timeout")?;

    stream
        .write_all(args.request_json.trim_end().as_bytes())
        .context("failed to write daemon request")?;
    stream
        .write_all(b"\n")
        .context("failed to terminate daemon request")?;
    stream.flush().context("failed to flush daemon request")?;

    let mut response = String::new();
    let bytes = BufReader::new(stream)
        .read_line(&mut response)
        .context("failed to read daemon response")?;
    if bytes == 0 {
        bail!("daemon closed the connection without a response")
    }
    serde_json::from_str::<Value>(response.trim_end()).context("daemon returned invalid JSON")?;
    println!("{}", response.trim_end());
    Ok(())
}

fn enrich_response(
    response: &str,
    method: &str,
    project: &Path,
    socket: &Path,
    worker_pid: u32,
    started_at_unix_ms: u64,
) -> Result<String> {
    let mut response_value: Value =
        serde_json::from_str(response.trim_end()).context("Python worker returned invalid JSON")?;
    if matches!(method, "ping" | "status")
        && let Some(result) = response_value
            .get_mut("result")
            .and_then(Value::as_object_mut)
    {
        result.insert("daemon_pid".to_string(), Value::from(std::process::id()));
        result.insert("worker_pid".to_string(), Value::from(worker_pid));
        result.insert(
            "project".to_string(),
            Value::from(project.to_string_lossy().into_owned()),
        );
        result.insert(
            "socket".to_string(),
            Value::from(socket.to_string_lossy().into_owned()),
        );
        result.insert(
            "started_at_unix_ms".to_string(),
            Value::from(started_at_unix_ms),
        );
    }
    serde_json::to_string(&response_value).context("failed to serialize daemon response")
}

fn error_response(id: Value, code: &str, message: impl Into<String>) -> Value {
    json!({
        "id": id,
        "ok": false,
        "error": {
            "code": code,
            "message": message.into(),
        }
    })
}

fn write_response(stream: &mut UnixStream, response: &str) -> Result<()> {
    stream
        .write_all(response.as_bytes())
        .context("failed to write daemon response")?;
    stream
        .write_all(b"\n")
        .context("failed to terminate daemon response")?;
    stream.flush().context("failed to flush daemon response")?;
    Ok(())
}

fn write_state_files(
    args: &ServeArgs,
    worker_pid: Option<u32>,
    started_at_unix_ms: u64,
) -> Result<()> {
    atomic_write(
        &args.pid_file,
        format!("{}\n", std::process::id()).as_bytes(),
    )?;
    let metadata = DaemonMetadata {
        version: 1,
        pid: std::process::id(),
        worker_pid,
        project: &args.project,
        socket: &args.socket,
        started_at_unix_ms,
    };
    let metadata_json =
        serde_json::to_vec_pretty(&metadata).context("failed to serialize daemon metadata")?;
    atomic_write(&args.metadata_file, &metadata_json)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("state file path has no valid file name: {}", path.display()))?;
    let temporary_path = path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));
    fs::write(&temporary_path, contents).with_context(|| {
        format!(
            "failed to write temporary state file {}",
            temporary_path.display()
        )
    })?;
    fs::rename(&temporary_path, path)
        .with_context(|| format!("failed to publish state file {}", path.display()))?;
    Ok(())
}

fn ensure_parent_directory(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("path has no parent directory: {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create directory {}", parent.display()))
}

fn remove_if_exists(path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => eprintln!(
            "failed to remove stale state file {}: {error}",
            path.display()
        ),
    }
}
